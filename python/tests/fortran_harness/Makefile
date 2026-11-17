# Element-level parity harness: the unmodified RF1 element routines from mis/
# (EMG, SMA1 and SDR2 paths for CROD/CONROD, CSHEAR, CTRMEM, CQDMEM, CQDMEM1,
# CQDMEM2, plus the CTRMEM thermal load TRIMEM) linked against GINO/executive
# stubs and a driver that fills the EST/ECPT record and prints the element matrices and stress/force words.
ifeq ($(origin FC),default)
FC      := gfortran
endif
FFLAGS  ?= -O0 -std=legacy -fno-automatic -finit-local-zero -w
MIS     := ../../../mis
MDS     := ../../../mds
KERNELS := ../fortran_kernels
BUILD   := build

ELEMENTS := krod rods rodd shears sheard ktrmem trmems trmemd kqdmem ktriqd \
            qdmm1s qdmm1d qdmm2s qdmm2d
STRESS   := srod1 srod2 spanl1 spanl2 strme1 sqdme1 sqdm11 sqdm12 sqdm21 sqdm22 \
            stqme2
LOADS    := trimem
SUPPORT  := premat pretrs pretrd gmmats gmmatd emastq emadtq ektrms ektrmd \
            q2trms q2bcs q2trmd q2bcd smmats saxb sadotb dadotb daxb invers \
            inverd sdrchk sort mpyl basglb

OBJECTS := $(addprefix $(BUILD)/,$(addsuffix .o,$(ELEMENTS) $(STRESS) $(LOADS) $(SUPPORT))) \
           $(BUILD)/khrfn1.o $(BUILD)/kstubs.o $(BUILD)/stubs.o $(BUILD)/driver.o

all: $(BUILD)/harness

$(BUILD):
	mkdir -p $(BUILD)

$(BUILD)/%.o: $(MIS)/%.f | $(BUILD)
	$(FC) $(FFLAGS) -c $< -o $@

$(BUILD)/khrfn1.o: $(MDS)/khrfn1.f | $(BUILD)
	$(FC) $(FFLAGS) -c $< -o $@

$(BUILD)/kstubs.o: $(KERNELS)/stubs.f | $(BUILD)
	$(FC) $(FFLAGS) -c $< -o $@

$(BUILD)/%.o: %.f | $(BUILD)
	$(FC) $(FFLAGS) -c $< -o $@

$(BUILD)/harness: $(OBJECTS)
	$(FC) $(FFLAGS) $^ -o $@

clean:
	rm -rf $(BUILD)

.PHONY: all clean

# Rod thermal/deformation loads (mis/rod.f as called from EDTL): a separate
# binary so the element driver stays untouched. BASGLB's CSTM read is renamed
# to the CSTMRD stub in loadstubs.f.
LOAD_OBJECTS := $(addprefix $(BUILD)/,rod.o basglb_cs.o mpyl.o premat.o sort.o gmmats.o \
                khrfn1.o kstubs.o loadstubs.o rodload.o)

all: $(BUILD)/rodload

$(BUILD)/basglb_cs.o: $(MIS)/basglb.f | $(BUILD)
	$(FC) $(FFLAGS) -c $< -o $@
	objcopy --redefine-sym read_=cstmrd_ $@

$(BUILD)/rodload: $(LOAD_OBJECTS)
	$(FC) $(FFLAGS) $^ -o $@

# CQDMEM1 thermal load (mis/qdmm1.f as called from EDTL): a separate binary;
# BASGLB is fed through its /TRANX/ cache (single non-basic system per element).
QDLOAD_OBJECTS := $(addprefix $(BUILD)/,qdmm1.o basglb.o premat.o gmmats.o mpyl.o sort.o \
                  khrfn1.o kstubs.o stubs.o qdmm1load.o)

all: $(BUILD)/qdmm1load

$(BUILD)/qdmm1load: $(QDLOAD_OBJECTS)
	$(FC) $(FFLAGS) $^ -o $@
