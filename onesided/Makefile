CXX       = mpic++
EXEC      = mpi_onesided

all: $(EXEC)

CXXFLAGS  = 


OBJS1     = mpi_onesided.o
LIBS1     = 

mpi_onesided: $(OBJS1)
	$(CXX) $(OBJS1) $(LIBS1) -o $@

.PHONY: clean
clean :
	rm -f $(EXEC) $(OBJS1)
