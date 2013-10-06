OBJS=conv2d.o readjpeg.o

%.o: %.cpp readjpeg.h
	g++ -O2 -c $<

all: $(OBJS)
	g++ $(OBJS) -o conv2d -ljpeg