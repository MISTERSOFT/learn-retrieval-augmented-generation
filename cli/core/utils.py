import json
from pathlib import Path

def load_file(file_path: str, file_format: str):
    file_content = None
    try:
        with Path(file_path).open() as file:
            match file_format:
                case 'json':
                    file_content = json.load(file)
                case 'txt':
                    file_content = file.read()
                case _:
                    pass
    except FileNotFoundError | json.decoder.JSONDecodeError:
        print(f"Error: '{file_path}' file not found.")
    finally:
        return file_content
