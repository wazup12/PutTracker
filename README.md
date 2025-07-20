# Put Strike Analyzer

This tool analyzes options trading data, focusing on sold PUT options to compute remaining income potential and detect spread strategies.

## Features

- Analyze sold PUTs for remaining income.
- Detect credit spread strategies.
- Interactive TUI for sorting and filtering.

## Usage

To run the tool, execute the following command. If you provide a file path directly, the tool will open it. Otherwise, a file selector will be shown.

```bash
python main.py [optional_file_path]
```

### File Selector Interactions

When the file selector is active, you can interact with it in the following ways:

- **Filtering:** Start typing in the input box to dynamically filter the list of files in the current directory. The list updates in real-time.
- **Autocomplete:** Press the `Tab` key to autocomplete the file name based on your current input. If there are multiple matches, pressing `Tab` repeatedly will cycle through them.
- **Highlighting:**
    - The currently selected item is highlighted with a dark grey background.
    - An item that is an *exact match* to your typed input or the autocompleted name will be highlighted with a dark green background.
- **Selection:**
    - Press `Enter` to load the currently highlighted file and proceed to the analysis screen. This works whether you are focused on the input box or the list itself.
    - **Click** on any file name in the list to load it immediately.