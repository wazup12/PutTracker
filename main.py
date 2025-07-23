import argparse
import configparser
import pandas as pd
from config_manager import ConfigManager
from dataframe_utils import (
    ensure_dataframe,
    parse_symbol,
    filter_options,
    drop_invalid_options,
    clean_price_column,
)
import logging
from datetime import datetime
import os
import sys
import io

from textual.app import App, ComposeResult
from textual.screen import Screen
from textual.widgets import Button, DataTable, Footer, Header, Input, Label
from textual.reactive import reactive
from textual.binding import Binding
from typing import cast


def _read_file_skip_comments(file_path):
    """Reads a file, skipping lines until the header line (containing 'Symbol') is encountered.
    Returns the content from the header line onwards.
    """
    with open(file_path, "r") as f:
        lines = f.readlines()

    for i, line in enumerate(lines):
        if "Symbol" in line:
            return "".join(lines[i:])
    return "".join(lines)


def get_sold_puts_data(file_path, logger) -> pd.DataFrame:
    """Returns a DataFrame of sold puts."""
    try:
        file_content = _read_file_skip_comments(file_path)
        df = pd.read_csv(io.StringIO(file_content), sep=",", quotechar='"')
        df.columns = [col.split("(")[0].strip() for col in df.columns]
        # Use utility functions for option filtering and cleaning
        options_df = filter_options(df)
        options_df = ensure_dataframe(options_df)
        options_df = parse_symbol(options_df, logger)
        options_df["Qty"] = pd.to_numeric(options_df["Qty"], errors="coerce")
        options_df = drop_invalid_options(options_df)
        sold_puts = options_df[
            (options_df["Type"] == "P") & (options_df["Qty"] < 0)
        ].copy()
        sold_puts = ensure_dataframe(sold_puts)
        sold_puts = clean_price_column(sold_puts, "Price")
        sold_puts.dropna(subset=["Price"], inplace=True)
        sold_puts["Days Until Expiration"] = (
            sold_puts["Expiration"] - datetime.now()
        ).dt.days
        sold_puts = sold_puts[sold_puts["Days Until Expiration"] > 0]
        sold_puts = ensure_dataframe(sold_puts)
        sold_puts["Remaining Income"] = 100 * sold_puts["Price"] / sold_puts["Strike"]
        sold_puts["Remaining % per Day"] = (
            sold_puts["Remaining Income"] / sold_puts["Days Until Expiration"]
        )
        output_df = ensure_dataframe(
            sold_puts[
                [
                    "Ticker",
                    "Expiration",
                    "Strike",
                    "Qty",
                    "Price",
                    "Remaining Income",
                    "Days Until Expiration",
                    "Remaining % per Day",
                ]
            ].copy()
        )
        output_df = output_df.rename(
            columns={"Price": "Current Price", "Remaining Income": "Remaining %"}
        )
        output_df = output_df.sort_values(by="Remaining % per Day", ascending=False)
        output_df["Expiration"] = output_df["Expiration"].dt.strftime("%m/%d/%Y")
        output_df["Remaining %"] = output_df["Remaining %"].map("{:.2f}%".format)
        output_df["Remaining % per Day"] = output_df["Remaining % per Day"].map(
            "{:.2f}%".format
        )
        return output_df
    except FileNotFoundError:
        return pd.DataFrame()
    except Exception as e:
        return pd.DataFrame()


def get_spreads_data(file_path, logger) -> pd.DataFrame:
    """Returns a DataFrame of spreads."""
    try:
        file_content = _read_file_skip_comments(file_path)
        df = pd.read_csv(io.StringIO(file_content), sep=",", quotechar='"')
        df.columns = [col.split("(")[0].strip() for col in df.columns]
        options_df = filter_options(df)
        options_df = ensure_dataframe(options_df)
        options_df = parse_symbol(options_df, logger)
        options_df = drop_invalid_options(options_df)
        puts = options_df[(options_df["Type"] == "P")].copy()
        if isinstance(puts, pd.Series):
            puts = puts.to_frame().T
        puts = clean_price_column(puts, "Price")
        puts["Qty"] = pd.to_numeric(puts["Qty"], errors="coerce")
        puts.dropna(subset=["Price"], inplace=True)
        spreads = []
        for _, group in puts.groupby(["Ticker", "Expiration"]):
            if isinstance(group, pd.Series):
                group = group.to_frame().T
            if len(group) > 1:
                group["Qty"] = pd.to_numeric(group["Qty"], errors="coerce")
                short_puts = group[group["Qty"] < 0]
                long_puts = group[group["Qty"] > 0]
                for _, short_put in short_puts.iterrows():
                    for _, long_put in long_puts.iterrows():
                        if short_put["Qty"] == -long_put["Qty"]:
                            price_gap = short_put["Price"] - long_put["Price"]
                            strike_gap = short_put["Strike"] - long_put["Strike"]
                            if strike_gap > 0:
                                days_to_expiration = (
                                    short_put["Expiration"] - datetime.now()
                                ).days
                                if days_to_expiration > 0:
                                    expected_income = 100 * price_gap / strike_gap
                                    expected_income_per_day = (
                                        expected_income / days_to_expiration
                                    )
                                    risk_per_contract = strike_gap * 100
                                    spreads.append(
                                        {
                                            "Ticker": short_put["Ticker"],
                                            "Days Left": days_to_expiration,
                                            "Shrt Strike": short_put["Strike"],
                                            "Lng Strike": long_put["Strike"],
                                            "Shrt Price": short_put["Price"],
                                            "Lng Price": long_put["Price"],
                                            "Price Gap": f"{price_gap:.2f}",
                                            "Strike Gap": strike_gap,
                                            "Exp. Income": f"{expected_income:.2f}%",
                                            "Exp. % per Day": f"{expected_income_per_day:.2f}%",
                                            "Risk/Contract": f"${risk_per_contract:,.2f}",
                                        }
                                    )
        if spreads:
            df_spreads = pd.DataFrame(spreads)
            return df_spreads
        return pd.DataFrame()
    except Exception as e:
        return pd.DataFrame()


from textual.containers import Vertical
from textual.widgets import ListView, ListItem


# get_config now imported from config_manager


from textual.containers import Center, Vertical


class UpdateDirectoryScreen(Screen):
    """Screen for updating the file directory."""

    CSS_PATH = "main.css"

    def compose(self) -> ComposeResult:
        app = cast(PutTracker, self.app)
        yield Header()
        with Center():
            with Vertical(id="update_directory_container"):
                yield Label("Enter new directory path:")
                if sys.platform == "win32":
                    yield Input(
                        placeholder="e.g., C:\\Users\\YourUser\\Documents\\data",
                        id="directory_input",
                    )
                else:
                    yield Input(
                        placeholder="e.g., ~/Documents/data", id="directory_input"
                    )
                if app.file_directory:
                    yield Label(
                        f"Current directory: {app.file_directory}",
                        classes="current-directory-label",
                    )
        yield Footer()

    def on_mount(self) -> None:
        """Focus the input on mount."""
        self.query_one(Input).focus()

    def on_input_submitted(self, event: Input.Submitted) -> None:
        """Handle submission of the new directory path."""
        app = cast(PutTracker, self.app)
        new_path = event.value
        app.config_manager.set_file_directory(new_path)
        app.file_directory = app.config_manager.get_file_directory()
        if app.file_path:
            app.switch_screen(MainScreen())
        else:
            app.switch_screen(FileSelectorScreen())


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
        app = cast(PutTracker, self.app)
        yield Header()
        yield Vertical(
            Label("Select a file:"),
            Input(
                placeholder="Start typing to filter files...", id="file_filter_input"
            ),
            ListView(id="file_list"),
            Label("Press Ctrl+U to update directory", id="hint_label"),
        ).add_class("file-selector-container")
        yield Footer()

    def on_mount(self) -> None:
        app = cast(PutTracker, self.app)
        self.refresh_file_list()

    def on_screen_resume(self) -> None:
        app = cast(PutTracker, self.app)
        self.refresh_file_list()

    def refresh_file_list(self) -> None:
        app = cast(PutTracker, self.app)
        self._autocomplete_base_value = ""
        try:
            self._current_autocomplete_matches = [
                entry.name
                for entry in os.scandir(app.file_directory)
                if entry.is_file()
            ]
        except FileNotFoundError:
            self._current_autocomplete_matches = []
        self._autocomplete_cycle_index = -1
        self.update_file_list("")
        self.query_one("#file_filter_input", Input).value = ""

    def on_input_changed(self, event: Input.Changed) -> None:
        app = cast(PutTracker, self.app)
        if self._is_programmatic_change:
            self._is_programmatic_change = False
            self.update_file_list(event.value)
            return

        if not self._is_autocompleting:
            self._autocomplete_base_value = event.value
            self._autocomplete_cycle_index = -1
            try:
                self._current_autocomplete_matches = [
                    entry.name
                    for entry in os.scandir(app.file_directory)
                    if entry.is_file()
                    and entry.name.lower().startswith(
                        self._autocomplete_base_value.lower()
                    )
                ]
            except FileNotFoundError:
                self._current_autocomplete_matches = []

        self._is_autocompleting = False
        self.update_file_list(event.value)

    def on_input_submitted(self, event: Input.Submitted) -> None:
        app = cast(PutTracker, self.app)
        """Handle the user pressing enter in the input box."""
        self.action_load_file()

    def on_list_view_highlighted(self, event: ListView.Highlighted) -> None:
        pass

    def update_file_list(self, filter_str: str):
        app = cast(PutTracker, self.app)
        list_view = self.query_one("#file_list", ListView)
        file_input = self.query_one("#file_filter_input", Input)
        list_view.clear()

        effective_filter_str = (
            self._autocomplete_base_value if self._is_autocompleting else filter_str
        )

        highlight_index = -1
        current_index = 0

        try:
            for entry in os.scandir(app.file_directory):
                if entry.is_file() and entry.name.lower().startswith(
                    effective_filter_str.lower()
                ):
                    item = ListItem(Label(entry.name))
                    if (
                        entry.name == file_input.value
                        or entry.name == self._autocomplete_base_value
                    ):
                        item.add_class("explicit-match")

                    list_view.append(item)
                    if entry.name == file_input.value:
                        highlight_index = current_index
                    current_index += 1
        except (OSError, FileNotFoundError):
            pass  # Ignore errors

        if highlight_index != -1:
            list_view.highlighted_index = highlight_index  # type: ignore

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

        self._autocomplete_cycle_index = (self._autocomplete_cycle_index + 1) % (
            len(self._current_autocomplete_matches) + 1
        )

        if self._autocomplete_cycle_index == len(self._current_autocomplete_matches):
            # Cycle back to the original prefix
            self._is_programmatic_change = True
            file_input.value = self._autocomplete_base_value
            self.update_file_list(file_input.value)
        else:
            # Autocomplete to the next match
            self._is_programmatic_change = True
            file_input.value = self._current_autocomplete_matches[
                self._autocomplete_cycle_index
            ]
            self.update_file_list(file_input.value)

    def action_load_file(self) -> None:
        app = cast(PutTracker, self.app)
        """Load the selected file when Enter is pressed."""
        list_view = self.query_one(ListView)
        file_name = None

        # Prioritize the highlighted item in the list
        if list_view.highlighted_child:
            file_name = list_view.highlighted_child.children[0].renderable  # type: ignore
        else:
            # Fallback to the input value if nothing is highlighted
            file_input = self.query_one(Input)
            file_name = file_input.value

        if file_name:
            file_path = os.path.join(app.file_directory, str(file_name))
            if os.path.isfile(file_path):
                app.file_path = file_path
                app.switch_screen(MainScreen())
            else:
                # If the file is not found, try to complete the path with the base directory
                completed_path = os.path.join(app.file_directory, file_name)
                if os.path.isfile(completed_path):
                    app.file_path = completed_path
                    app.switch_screen(MainScreen())

    def action_update_config(self) -> None:
        """Switch to the update directory screen."""
        self.app.push_screen(UpdateDirectoryScreen())


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
        app = cast(PutTracker, self.app)
        file_path = app.file_path
        app_logger = app.app_logger
        app_logger.info(f"[TUI] on_mount: file_path={file_path}")
        sold_puts_data = get_sold_puts_data(file_path, app_logger)
        spreads_data = get_spreads_data(file_path, app_logger)

        app_logger.info(
            f"[TUI] sold_puts_data shape: {sold_puts_data.shape}, columns: {sold_puts_data.columns.tolist()}"
        )
        app_logger.info(
            f"[TUI] spreads_data shape: {spreads_data.shape}, columns: {spreads_data.columns.tolist()}"
        )

        sold_puts_table = self.query_one("#sold_puts_table")
        app_logger.info(f"[TUI] Got sold_puts_table widget: {sold_puts_table}")
        if not sold_puts_data.empty:
            app_logger.info("[TUI] Adding columns and rows to sold_puts_table")
            sold_puts_table.add_columns(*sold_puts_data.columns.to_list())  # type: ignore
            sold_puts_table.add_rows(sold_puts_data.values.tolist())  # type: ignore
            app_logger.info("[TUI] Added columns and rows to sold_puts_table")
        else:
            app_logger.info("[TUI] sold_puts_data is empty, not adding to table")

        spreads_table = self.query_one("#spreads_table")
        app_logger.info(f"[TUI] Got spreads_table widget: {spreads_table}")
        if not spreads_data.empty:
            app_logger.info("[TUI] Adding columns and rows to spreads_table")
            spreads_table.add_columns(*spreads_data.columns.to_list())  # type: ignore
            spreads_table.add_rows(spreads_data.values.tolist())  # type: ignore
            app_logger.info("[TUI] Added columns and rows to spreads_table")
        else:
            app_logger.info("[TUI] spreads_data is empty, not adding to table")


class PutTracker(App):
    """A Textual app to analyze put strikes."""

    file_directory: str
    file_path: str | None
    config: configparser.ConfigParser
    app_logger: logging.Logger

    def __init__(self, file_path=None):
        super().__init__()
        self.config_missing = not os.path.exists("config.ini")
        self.config_manager = ConfigManager()
        self.config = self.config_manager.config
        self.file_directory = self.config_manager.get_file_directory()
        self.file_path = file_path

        # Set up logging level based on DEBUG in config
        debug_flag = self.config["DEFAULT"].get("DEBUG", "false").lower() == "true"
        log_level = logging.DEBUG if debug_flag else logging.WARNING
        if debug_flag:
            logging.basicConfig(
                filename="debug.log",
                level=log_level,
                format="%(asctime)s - %(levelname)s - %(message)s",
            )
        else:
            logging.basicConfig(
                level=log_level, format="%(asctime)s - %(levelname)s - %(message)s"
            )
        self.app_logger = logging.getLogger(__name__)

    def on_mount(self) -> None:
        if self.config_missing:
            self.push_screen(UpdateDirectoryScreen())
        elif self.file_path:
            self.push_screen(MainScreen())
        else:
            self.push_screen(FileSelectorScreen())

    def on_key(self, event) -> None:
        if event.key == "ctrl+u":
            self.action_update_config()

    def action_update_config(self) -> None:
        """Switch to the update directory screen."""
        self.push_screen(UpdateDirectoryScreen())


def main():
    parser = argparse.ArgumentParser(description="Put Strike Analyzer")
    parser.add_argument("file", nargs="?", help="Path to the TSV file")
    args = parser.parse_args()

    app = PutTracker(file_path=args.file)
    app.run()


if __name__ == "__main__":
    main()
