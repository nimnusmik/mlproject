import sys 
import logging


def error_message_detail(error, error_detail: sys):
    _, _, exc_tb = error_detail.exc_info() #3values

    file_name = exc_tb.tb_frame.f_code.co_filename
    line_number = exc_tb.tb_frame.f_lineno

    # Get the name of the function where the error occurred
    error_message = f"Error occurred in script: [{file_name}] at line number: [{line_number}] error message: [{str(error)}]"
    
    return error_message

class CustomException(Exception):
    def __init__(self, error_message, error_deatail: sys):
        super().__init__(error_message)
        self.error_message = error_message_detail(error_message, error_deatail=error_deatail)

        def __str__(self):
            return self.error_message
        
if __name__ == "__main__":

    try:
        a = 1/0
    except Exception as e:
        logging.info("Divide by Zero.")
        raise CustomException(e, sys)