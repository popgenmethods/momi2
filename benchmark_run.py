from subprocess import call

out_file = open("benchmark.out",'a',0)

def do_runs(n_total_log2):
    for n_demes_log2 in range(1, n_total_log2+1):
        n_demes = 2**n_demes_log2
        n_per_deme = 2**(n_total_log2 - n_demes_log2)
        cmd = "./benchmark.py %d %d %d %d" % (n_demes, n_per_deme, reps, cores) 
        out_file.write(cmd + "\n")

        call(cmd, shell=True, stdout=out_file, stderr=out_file)

cores = 20
reps = 20
#for i in range(1, 9):
#    do_runs(i)

cores=5
for n_demes in [32,64,128,256]:
    n_per_deme = int(256/n_demes)
    cmd = "./benchmark.py %d %d %d %d --moranOnly" % (n_demes, n_per_deme, reps, cores) 
    out_file.write(cmd + "\n")

    call(cmd, shell=True, stdout=out_file, stderr=out_file)
