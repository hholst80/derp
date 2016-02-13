FC=gfortran
FOPTS=-O2 -fbounds-check -Wall

derp: derp.f
	$(FC) $(FOPTS) -o $@ $<
