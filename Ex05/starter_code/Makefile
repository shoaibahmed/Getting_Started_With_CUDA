NET=main.o network.o train.o
AUX=helper.o print_h.o read_file.o arguments.o s_batch.o
OBJ=$(NET) $(AUX)
INC=protos.h net_include.h
CFLAGS=-O

LIB=-lm

neuron: $(OBJ) $(INC)
	$(CC) $(CFLAGS) -o neuron $(OBJ) $(LIB)

.c.o: $(INC)
	$(CC) -c $(CFLAGS) $(CPPFLAGS) -o $@ $<
