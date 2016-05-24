//
// An example using MPI one-sided communications
//
#include <iostream>
#include <cstdio>
#include <cstdlib>
#include <cassert>
#include <mpi.h>

using namespace std;

int main(int argc, char* argv[])
{
  int rank, n;
  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &n);

  // All nodes send one float (nsend=1) to node 0 (target=0)
  const int target= 0;
  const int nsend=1;

  MPI_Win win_header, win_data;
  
  float* dat;
  MPI_Win_allocate_shared(sizeof(float)*n, sizeof(float), MPI_INFO_NULL,
			  MPI_COMM_WORLD, &dat, &win_data);
  
  // header is the number of data already written
  int ndat= 0;

  MPI_Win_create(&ndat, sizeof(int), sizeof(int), MPI_INFO_NULL,
		 MPI_COMM_WORLD, &win_header);

  // start accepting access to data
  MPI_Win_fence(0, win_data);

  int offset;
  float send= rank + 1;
  // each node is going to send a float send=rank to node 'target'

  // Get offset
  MPI_Win_lock(MPI_LOCK_EXCLUSIVE, target, 0, win_header);
  // offset = target::ndat
  // target::ndat += nsend
  MPI_Get_accumulate(&nsend, 1, MPI_INT, &offset, 1, MPI_INT, target, 0, 1,
		     MPI_INT, MPI_SUM, win_header);
  MPI_Win_unlock(target, win_header);

  MPI_Put(&send, 1, MPI_FLOAT, target, offset, 1, MPI_FLOAT, win_data);

  MPI_Win_fence(0, win_data);
  
  if(rank == target) {
    printf("ndat after communication: %d\n", ndat);
    for(int i=0; i<ndat; ++i)
      printf("received data[%d] = %f\n", i, dat[i]);
  }

  MPI_Win_free(&win_header);
  MPI_Win_free(&win_data);

  MPI_Finalize();

  return 0;
}
