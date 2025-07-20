import argparse
import pandas as pd
from datetime import datetime
import os

from textual.app import App, ComposeResult
from textual.screen import Screen
from textual.widgets import Button, DataTable, Footer, Header, Input, Label
from textual.reactive import reactive
from textual.binding import Binding

def parse_symbol(df):
    """Extracts Ticker, Expiration, Strike, and Type from the 'Symbol' column."""
    try:
        symbol_parts = df['Symbol'].str.split(' ', expand=True)
        df['Ticker'] = symbol_parts[0]
        df['Expiration'] = pd.to_datetime(symbol_parts[1], format='%m/%d/%Y')
        df['Strike'] = pd.to_numeric(symbol_parts[2])
        df['Type'] = symbol_parts[3]
    except Exception as e:
        # Handle cases where the symbol format is unexpected
        print(f"Warning: Could not parse all symbols. Error: {e}")
        df[['Ticker', 'Expiration', 'Strike', 'Type']] = [None, pd.NaT, None, None]
    return df

def get_sold_puts_data(file_path):
    """Returns a DataFrame of sold puts."""
    try:
        df = pd.read_csv(file_path, sep='\t')
        df.columns = [col.split('(')[0].strip() for col in df.columns]
        df = parse_symbol(df)

        # Filter out rows that couldn't be parsed
        df.dropna(subset=['Ticker', 'Expiration', 'Strike', 'Type'], inplace=True)

        sold_puts = df[(df['Security Type'] == 'Option') & (df['Type'] == 'P') & (df['Qty'] < 0)].copy()

        sold_puts['Price'] = pd.to_numeric(sold_puts['Price'].replace({r'\$': ''}, regex=True), errors='coerce')
        sold_puts.dropna(subset=['Price'], inplace=True)


        sold_puts['Days Until Expiration'] = (sold_puts['Expiration'] - datetime.now()).dt.days
        # Avoid division by zero or by days in the past
        sold_puts = sold_puts[sold_puts['Days Until Expiration'] > 0]

        sold_puts['Remaining Income'] = 100 * sold_puts['Price'] / sold_puts['Strike']
        sold_puts['Remaining % per Day'] = sold_puts['Remaining Income'] / sold_puts['Days Until Expiration']

        output_df = sold_puts[['Ticker', 'Expiration', 'Strike', 'Qty', 'Price', 'Remaining Income', 'Days Until Expiration', 'Remaining % per Day']]
        output_df = output_df.rename(columns={'Price': 'Current Price', 'Remaining Income': 'Remaining %'})
        output_df = output_df.sort_values(by='Remaining % per Day', ascending=False)

        output_df['Expiration'] = output_df['Expiration'].dt.strftime('%m/%d/%Y')
        output_df['Remaining %'] = output_df['Remaining %'].map('{:.2f}%'.format)
        output_df['Remaining % per Day'] = output_df['Remaining % per Day'].map('{:.2f}%'.format)

        return output_df

    except FileNotFoundError:
        return pd.DataFrame()
    except Exception:
        return pd.DataFrame()

def get_spreads_data(file_path):
    """Returns a DataFrame of spreads."""
    try:
        df = pd.read_csv(file_path, sep='\t')
        df.columns = [col.split('(')[0].strip() for col in df.columns]
        df = parse_symbol(df)
        df.dropna(subset=['Ticker', 'Expiration', 'Strike', 'Type'], inplace=True)


        puts = df[(df['Security Type'] == 'Option') & (df['Type'] == 'P')].copy()
        puts['Price'] = pd.to_numeric(puts['Price'].replace({r'\$': ''}, regex=True), errors='coerce')
        puts.dropna(subset=['Price'], inplace=True)


        spreads = []
        for _, group in puts.groupby(['Ticker', 'Expiration']):
            if len(group) > 1:
                short_puts = group[group['Qty'] < 0]
                long_puts = group[group['Qty'] > 0]

                for _, short in short_puts.iterrows():
                    for _, long in long_puts.iterrows():
                        if short['Qty'] == -long['Qty']:
                            price_gap = short['Price'] - long['Price']
                            strike_gap = short['Strike'] - long['Strike']
                            if strike_gap > 0:
                                days_to_expiration = (short['Expiration'] - datetime.now()).days
                                if days_to_expiration > 0:
                                    expected_income = 100 * price_gap / strike_gap
                                    expected_income_per_day = expected_income / days_to_expiration
                                    risk_per_contract = strike_gap * 100

                                    spreads.append({
                                        'Ticker': short['Ticker'],
                                        'Expiration': short['Expiration'].strftime('%m/%d/%Y'),
                                        'Short Strike': short['Strike'],
                                        'Long Strike': long['Strike'],
                                        'Short Price': short['Price'],
                                        'Long Price': long['Price'],
                                        'Price Gap': f"{price_gap:.2f}",
                                        'Strike Gap': strike_gap,
                                        'Expected Income (%)': f"{expected_income:.2f}%",
                                        'Days to Expiration': days_to_expiration,
                                        'Expected % per Day': f"{expected_income_per_day:.2f}%",
                                        'Risk per Contract': f"${risk_per_contract:,.2f}"
                                    })
        if spreads:
            return pd.DataFrame(spreads)
        return pd.DataFrame()

    except Exception:
        return pd.DataFrame()

from textual.containers import Vertical
from textual.widgets import ListView, ListItem

class FileSelectorScreen(Screen):
    """Screen to select a file with a dynamic preview."""
    CSS_PATH = "main.css"
    _filtered_files = reactive([])
    _autocomplete_base_value = reactive("")
    _current_autocomplete_matches = reactive([])
    _autocomplete_cycle_index = reactive(-1)
    _is_programmatic_change = reactive(False)
    _is_autocompleting = reactive(False)

    BINDINGS = [
        Binding("tab", "autocomplete", "Autocomplete", show=False),
        Binding("enter", "load_file", "Load File", show=False),
    ]

    def compose(self) -> ComposeResult:
        yield Header()
        yield Vertical(
            Label("Select a file:"),
            Input(placeholder="Start typing to filter files...", id="file_filter_input"),
            ListView(id="file_list")
        ).add_class("file-selector-container")
        yield Footer()

    def on_mount(self) -> None:
        self._autocomplete_base_value = ""
        self._current_autocomplete_matches = []
        self._autocomplete_cycle_index = -1
        self.update_file_list("")

    def on_input_changed(self, event: Input.Changed) -> None:
        if self._is_programmatic_change:
            self._is_programmatic_change = False
            self.update_file_list(event.value)
            return
        
        if not self._is_autocompleting:
            self._autocomplete_base_value = event.value
            self._autocomplete_cycle_index = -1
            self._current_autocomplete_matches = [
                entry.name for entry in os.scandir(".")
                if entry.is_file() and entry.name.lower().startswith(self._autocomplete_base_value.lower())
            ]
        
        self._is_autocompleting = False
        self.update_file_list(event.value)

    def on_list_view_selected(self, event: ListView.Selected) -> None:
        file_path = event.item.children[0].renderable
        if os.path.isfile(str(file_path)):
            self.parent.file_path = str(file_path)
            self.app.switch_screen(MainScreen())

    def on_list_view_highlighted(self, event: ListView.Highlighted) -> None:
        pass

    def update_file_list(self, filter_str: str):
        list_view = self.query_one("#file_list", ListView)
        file_input = self.query_one("#file_filter_input", Input)
        list_view.clear()
        
        effective_filter_str = self._autocomplete_base_value if self._is_autocompleting else filter_str
        
        highlight_index = -1
        current_index = 0

        try:
            for entry in os.scandir("."):
                if entry.is_file() and entry.name.lower().startswith(effective_filter_str.lower()):
                    list_view.append(ListItem(Label(entry.name)))
                    if entry.name == file_input.value:
                        highlight_index = current_index
                    current_index += 1
        except OSError:
            pass # Ignore errors
        
        if highlight_index != -1:
            list_view.highlighted_index = highlight_index

    def action_autocomplete(self) -> None:
        file_input = self.query_one("#file_filter_input", Input)
        self._is_autocompleting = True

        if not self._current_autocomplete_matches:
            # If no matches, just keep the current input as the base value
            self._autocomplete_base_value = file_input.value
            self._autocomplete_cycle_index = -1
            self._is_programmatic_change = True
            file_input.value = self._autocomplete_base_value
            self.update_file_list(file_input.value)
            return

        self._autocomplete_cycle_index = (self._autocomplete_cycle_index + 1) % (len(self._current_autocomplete_matches) + 1)

        if self._autocomplete_cycle_index == len(self._current_autocomplete_matches):
            # Cycle back to the original prefix
            self._is_programmatic_change = True
            file_input.value = self._autocomplete_base_value
            self.update_file_list(file_input.value)
        else:
            # Autocomplete to the next match
            self._is_programmatic_change = True
            file_input.value = self._current_autocomplete_matches[self._autocomplete_cycle_index]
            self.update_file_list(file_input.value)

    def action_load_file(self) -> None:
        file_input = self.query_one("#file_filter_input", Input)
        file_name = file_input.value
        if file_name:
            self.app.exit(str(file_name))


class MainScreen(Screen):
    """The main analysis screen with data tables."""
    CSS_PATH = "main.css"
    def compose(self) -> ComposeResult:
        yield Header()
        yield Label("Sold Puts Analysis")
        yield DataTable(id="sold_puts_table")
        yield Label("Spreads Analysis")
        yield DataTable(id="spreads_table")
        yield Footer()

    def on_mount(self) -> None:
        file_path = self.parent.file_path
        sold_puts_data = get_sold_puts_data(file_path)
        spreads_data = get_spreads_data(file_path)

        sold_puts_table = self.query_one("#sold_puts_table")
        if not sold_puts_data.empty:
            sold_puts_table.add_columns(*sold_puts_data.columns.to_list())
            sold_puts_table.add_rows(sold_puts_data.values.tolist())

        spreads_table = self.query_one("#spreads_table")
        if not spreads_data.empty:
            spreads_table.add_columns(*spreads_data.columns.to_list())
            spreads_table.add_rows(spreads_data.values.tolist())


class PutStrikeAnalyzerApp(App):
    """A Textual app to analyze put strikes."""
    def __init__(self, file_path=None):
        super().__init__()
        self.file_path = file_path

    def on_mount(self) -> None:
        if self.file_path:
            self.push_screen(MainScreen())
        else:
            self.push_screen(FileSelectorScreen())


def main():
    parser = argparse.ArgumentParser(description="Put Strike Analyzer")
    parser.add_argument("file", nargs="?", help="Path to the TSV file")
    args = parser.parse_args()

    app = PutStrikeAnalyzerApp(file_path=args.file)
    app.run()

if __name__ == "__main__":
    main()