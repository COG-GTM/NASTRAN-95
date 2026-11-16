# gfortran build of NASTRAN-95 (Linux x86-64).
# Sources are copied to build/src with the DOS end-of-file marker (Ctrl-Z) removed;
# the NASA sources in mds/ mis/ bd/ are otherwise compiled as delivered.
FC      := gfortran
FFLAGS  ?= -std=legacy -fno-automatic -fallow-argument-mismatch -fallow-invalid-boz \
           -fno-range-check -w
LDFLAGS ?=

B       := build
SRCS    := $(wildcard mds/*.f mis/*.f bd/*.f)
INCS    := $(wildcard mds/*.COM mis/*.COM)
LIBSRC  := $(filter-out mds/nastrn.f mds/chkfil.f bd/%,$(SRCS))
BDSRC   := $(wildcard bd/*.f)
OBJ      = $(B)/obj/$(notdir $(1:.f=.o))
LIBOBJ  := $(foreach f,$(LIBSRC),$(call OBJ,$(f)))
BDOBJ   := $(foreach f,$(BDSRC),$(call OBJ,$(f)))
FIXED   := $(addprefix $(B)/src/,$(notdir $(INCS)))

.PHONY: all clean
RFSRC   := $(filter-out rf/NASINFO,$(wildcard rf/*)) rf/NASINFO
RF      := $(addprefix $(B)/rf/,$(notdir $(RFSRC)))

all: bin/nastran.exe bin/chkfil $(RF)

# Rigid format (DMAP) files, cleaned of CR and Ctrl-Z for Unix reads.
$(B)/rf/%: rf/%
	@mkdir -p $(B)/rf
	tr -d '\r\032' < $< > $@

$(B)/src/%: | $(B)/src
	@:
$(B)/src $(B)/obj:
	mkdir -p $@

define FIX_RULE
$(B)/src/$(notdir $(1)): $(1) | $(B)/src
	tr -d '\032' < $$< > $$@
endef
$(foreach f,$(SRCS) $(INCS),$(eval $(call FIX_RULE,$(f))))

$(B)/obj/%.o: $(B)/src/%.f $(FIXED) | $(B)/obj
	$(FC) $(FFLAGS) -I$(B)/src -c $< -o $@

$(B)/libnastran.a: $(LIBOBJ)
	rm -f $@ && ar rcs $@ $^

bin/nastran.exe: $(B)/obj/nastrn.o $(BDOBJ) $(B)/libnastran.a
	$(FC) $(LDFLAGS) -o $@ $^

bin/chkfil: $(B)/obj/chkfil.o
	$(FC) -o $@ $^

clean:
	rm -rf $(B) bin/nastran.exe bin/chkfil
