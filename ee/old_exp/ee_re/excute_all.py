import subprocess


scripts = [
"/home/haselab/Documents/tat/Research/app/pbl/main_multi_ds.py",
"/home/haselab/Documents/tat/Research/app/pbl/main_caltech.py",
]

for script in scripts:
    process = subprocess.Popen(['python', script], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    while True:
        output = process.stdout.readline()
        if output == '' and process.poll() is not None:
            break
        if output:
            print(output.strip())
    stderr = process.communicate()[1]
    if stderr:
        print(f"Error executing {script}: {stderr}")