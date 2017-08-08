CC = gcc
SRC = interp.c

libpppc.so:
	$(CC) -shared -fPIC -o $@ $(SRC)

.PHONY: clean

clean:
	rm -f *.o *.so libpppc.so
