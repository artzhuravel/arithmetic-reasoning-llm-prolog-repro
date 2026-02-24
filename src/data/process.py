from src.data.prepare_splits import prepare_splits

def process_data():
    gsm8k_prolog_train, openai_gsm8k_train, gsm8k_prolog_test, openai_gsm8k_test, gsm8k_prolog_val, openai_gsm8k_val = prepare_splits()
    return gsm8k_prolog_train, openai_gsm8k_train, gsm8k_prolog_test, openai_gsm8k_test, gsm8k_prolog_val, openai_gsm8k_val