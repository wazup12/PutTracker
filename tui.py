from textual.app import App, ComposeResult
from textual.widgets import Input, ListView, ListItem, Label
from textual.containers import Vertical
import os

class FilePreviewer(App):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._autocomplete_prefix = ""
        self._autocomplete_matches = []
        self._autocomplete_index = -1
        self._is_autocompleting = False

    def compose(self) -> ComposeResult:
        yield Vertical(
            Input(placeholder="Start typing to filter files...", id="file_input"),
            ListView(),
        )

    def on_mount(self) -> None:
        self.update_file_list("")

    def on_input_changed(self, event: Input.Changed) -> None:
        if self._is_autocompleting:
            self._is_autocompleting = False
        else:
            self._autocomplete_prefix = event.value
            self._autocomplete_index = -1
        self.update_file_list(event.value)

    def on_input_key(self, event) -> None:
        if event.key == "tab":
            event.prevent_default()
            input_widget = self.query_one(Input)

            if not self._autocomplete_matches:
                return

            if self._autocomplete_index == len(self._autocomplete_matches) - 1:
                self._autocomplete_index = -1
            else:
                self._autocomplete_index += 1

            self._is_autocompleting = True
            if self._autocomplete_index == -1:
                input_widget.value = self._autocomplete_prefix
            else:
                input_widget.value = self._autocomplete_matches[self._autocomplete_index]

    def update_file_list(self, filter_str: str):
        list_view = self.query_one(ListView)
        list_view.clear()
        self._autocomplete_matches = []
        try:
            for entry in os.scandir("."):
                if entry.name.startswith(filter_str):
                    self._autocomplete_matches.append(entry.name)
                    list_view.append(ListItem(Label(entry.name)))
        except OSError:
            pass # Ignore errors like permission denied

if __name__ == "__main__":
    app = FilePreviewer()
    app.run()
