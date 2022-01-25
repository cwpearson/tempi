print("in test.py")

print("import MPI...")
from mpi4py import MPI

print("MPI.COMM_WORLD")
comm = MPI.COMM_WORLD
print("comm.Get_rank()")
rank = comm.Get_rank()
print("got rank")

print(rank)

if rank == 0:
    data = {'a': 7, 'b': 3.14}
    comm.send(data, dest=1, tag=11)
elif rank == 1:
    data = comm.recv(source=0, tag=11)

print("end")