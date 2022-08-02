import sys

bcolors = {"blue": '\033[94m', "end": '\033[0m', "green": '\033[92m', "yellow": '\033[93m', "cyan": '\033[96m', "red": '\033[91m'}

def color_print(*args, color=None):
    if color is None:
        print(*args)
    else:
        print(bcolors[color], *args, bcolors["end"])

def error_print(*args):
    color_print(*args, color="red")

def warning_print(*args):
    color_print(*args, color="yellow")
        
def create_log_files(cfg, train_mode=True):
    assert(cfg.model_dir.output_dir is not None)

    if train_mode:
        info_logger_name = f"{cfg.model_dir.output_dir}/info.log"
        err_logger_name = f"{cfg.model_dir.output_dir}/err.log"
    else:
        info_logger_name = f"{cfg.model_dir.output_dir}/info_eval.log"
        err_logger_name = f"{cfg.model_dir.output_dir}/err_eval.log"
        
    return info_logger_name, err_logger_name

class LoggerManager():
    def __init__(self, cfg, train_mode=True):
        self.stdout_logger_name, self.stderr_logger_name = create_log_files(cfg, train_mode=train_mode)
        self.current_stdout_mode = "terminal"
        self.current_stderr_mode = "terminal"
        

    def redirect_to_stdout(self):
        if self.current_stdout_mode == "logger":
            sys.stdout.close()

        sys.stdout = sys.__stdout__
        self.current_stdout_mode = "terminal"

    def redirect_to_stderr(self):
        if self.current_stderr_mode == "logger":
            sys.stderr.close()
        sys.stderr = sys.__stderr__
        self.current_stderr_mode = "terminal"

    def redirect_stdout_to_logger(self):
        sys.stdout = open(self.stdout_logger_name, "w")
        self.current_stdout_mode = "logger"
        

    def redirect_stderr_to_logger(self):
        sys.stderr = open(self.stderr_logger_name, "w")
        self.current_stderr_mode = "logger"

    def use_std(self):
        self.redirect_to_stdout()
        self.redirect_to_stderr()

    def use_logger(self):
        self.redirect_stdout_to_logger()
        self.redirect_stderr_to_logger()

    def __del__(self):
        if self.current_stdout_mode == "logger":
            sys.stdout.close()
        if self.current_stderr_mode == "logger":
            sys.stderr.close()
        print("Closing logger!!!")
