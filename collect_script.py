import platform
import sys
import subprocess
import traceback

def get_os_info():
    return {
        'system': platform.system(),
        'node': platform.node(),
        'release': platform.release(),
        'version': platform.version(),
        'machine': platform.machine(),
        'processor': platform.processor(),
    }

def get_python_info():
    return {
        'implementation': platform.python_implementation(),
        'version': platform.python_version(),
        'compiler': platform.python_compiler(),
    }

def get_pip_list():
    try:
        result = subprocess.run([sys.executable, '-m', 'pip', 'list'], capture_output=True, text=True)
        if result.returncode == 0:
            return result.stdout
        else:
            return f"Failed to get pip list: {result.stderr}"
    except Exception as e:
        return f"An error occurred: {str(e)}"

def write_to_log_file(content, filename='env_output.log'):
    try:
        with open(filename, 'w') as file:
            file.write(content)
    except Exception as e:
        print(f"Error writing to file {filename}: {e}")
        traceback.print_exc()

def main():
    os_info = get_os_info()
    python_info = get_python_info()
    pip_list = get_pip_list()

    
    content = "Operating System Information:\n"
    for key, value in os_info.items():
        content += f"{key}: {value}\n"

    content += "\nPython Information:\n"
    for key, value in python_info.items():
        content += f"{key}: {value}\n"

    content += "\nPip List:\n"
    content += pip_list

    # stdout
    print(content)

    # file
    write_to_log_file(content)

if __name__ == "__main__":
    main()