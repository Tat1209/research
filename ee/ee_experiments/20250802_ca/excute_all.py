import subprocess

scripts = [
        # "/home/tat/research/ee/20250802_ca/main_p1.py",
        # "/home/tat/research/ee/20250802_ca/main_p2.py",
        "/home/tat/research/ee/20250802_ca/main_p3.py",
        "/home/tat/research/ee/20250802_ca/main_p4.py",
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