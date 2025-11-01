import subprocess


scripts = [
        "/home/haselab/Documents/tat/Research/app/ee_re_32/main_cifar_test.py",
        "/home/haselab/Documents/tat/Research/app/ee_re_32/main_cifar_test copy.py",
        "/home/haselab/Documents/tat/Research/app/ee_re_32/main_cifar_test copy 2.py",
        "/home/haselab/Documents/tat/Research/app/ee_re_32/main_cifar_test copy 3.py",
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