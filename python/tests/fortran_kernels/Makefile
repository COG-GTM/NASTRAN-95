# Builds the shared-kernel parity harnesses: the unmodified Fortran MAT (premat.f),
# TRANSS/TRANSD (pretrs.f, pretrd.f) and GMMATS/GMMATD linked against GINO stubs
# (build/kernels), and BASGLB/GLBBAS/GBTRAN (basglb.f, mpyl.f) (build/coords).
# kernels_temp runs MAT with TEMPID set, reading MATT1 and TABLEM1-4 from the stubs.
ifeq ($(origin FC),default)
FC      := gfortran
endif
FFLAGS  ?= -O0 -std=legacy -fno-automatic -finit-local-zero -w
MIS     := ../../../mis
BUILD   := build
SOURCES := $(MIS)/premat.f $(MIS)/pretrs.f $(MIS)/pretrd.f $(MIS)/gmmats.f $(MIS)/gmmatd.f
OBJECTS := $(patsubst $(MIS)/%.f,$(BUILD)/%.o,$(SOURCES)) $(BUILD)/stubs.o $(BUILD)/driver.o
COORDS  := $(BUILD)/basglb.o $(BUILD)/mpyl.o $(BUILD)/stubs_coords.o $(BUILD)/driver_coords.o

TEMP_OBJECTS := $(BUILD)/premat.o $(BUILD)/gmmats.o $(BUILD)/stubs_temp.o $(BUILD)/driver_temp.o

all: $(BUILD)/kernels $(BUILD)/coords $(BUILD)/kernels_temp

$(BUILD):
	mkdir -p $(BUILD)

$(BUILD)/%.o: $(MIS)/%.f | $(BUILD)
	$(FC) $(FFLAGS) -c $< -o $@

$(BUILD)/%.o: %.f | $(BUILD)
	$(FC) $(FFLAGS) -c $< -o $@

$(BUILD)/kernels: $(OBJECTS)
	$(FC) $(FFLAGS) $^ -o $@

$(BUILD)/coords: $(COORDS)
	$(FC) $(FFLAGS) $^ -o $@

$(BUILD)/kernels_temp: $(TEMP_OBJECTS)
	$(FC) $(FFLAGS) $^ -o $@

clean:
	rm -rf $(BUILD)

.PHONY: all clean
