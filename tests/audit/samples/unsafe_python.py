# Intentionally vulnerable Python sample for audit tests.
import pickle
import subprocess

PASSWORD = "hardcoded_secret"
API_TOKEN = "sk-1234567890abcdef"


def process(user_input):
    try:
        data = pickle.loads(user_input)
        result = eval(data)
        subprocess.call(result, shell=True)
    except:
        pass


def leaky():
    f = open("/tmp/audit_test.txt", "w")
    f.write("data")
