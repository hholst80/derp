FC=gfortran
FOPTS=-O2 -march=native -fbounds-check -Wall -Wno-unused-variable -Wno-unused-dummy-argument -fdefault-real-8

derp.exe: derp.f Makefile
	$(FC) $(FOPTS) -o $@ $<
