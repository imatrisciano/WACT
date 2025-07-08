class Colors:
    HEADER = '\033[95m'
    OK_BLUE = '\033[94m'
    OK_CYAN = '\033[96m'
    OK_GREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    END_COLOR = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

    @staticmethod
    def print_colored(text: str, colour):
        print(colour + text + Colors.END_COLOR)

    @staticmethod
    def set_color(color):
        print(color, end='')

    @staticmethod
    def reset():
        print(Colors.END_COLOR, end='')