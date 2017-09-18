CXX = g++

TVM_ROOT = tvm
NNVM_ROOT = nnvm
DMLC_CORE = dmlc-core


PKG_CFLAGS = -std=c++11 -O3 -g -fPIC\
	-Iinclude\
	-I${TVM_ROOT}/include\
	-I${DMLC_CORE}/include\
	-I${TVM_ROOT}/dlpack/include\
	-I${TVM_ROOT}/HalideIR/src

PKG_LDFLAGS =
UNAME_S := $(shell uname -s)


TARGET =
ifeq ($(UNAME_S), Darwin)
	TARGET = lib/libtvmflow.dylib
	PKG_LDFLAGS += -undefined dynamic_lookup
	WHOLE_ARCH= -all_load
	NO_WHOLE_ARCH= -noall_load
else
	TARGET = lib/libtvmflow.so
	WHOLE_ARCH= --whole-archive
	NO_WHOLE_ARCH= --no-whole-archive
endif


include $(DMLC_CORE)/make/dmlc.mk

ALL_DEP =

PKG_CFLAGS += -I${NNVM_ROOT}/include
ALL_DEP += ${DMLC_CORE}/libdmlc.a ${NNVM_ROOT}/lib/libnnvm.a


SRC = $(wildcard src/*/*/*.cc src/*/*.cc src/*.cc)
OBJ = $(patsubst %.cc, build/%.o, $(SRC))
ALL_DEP += ${OBJ}

#$(info $$SRC is [${SRC}])
#$(info $$OBJ is [${OBJ}])
#$(info $$ALL_DEP is [${ALL_DEP}])

.PHONY: clean all

all: ${TARGET}

nnvm/lib/libnnvm.a:
	+	cd nnvm; make ; cd -

$(DMLC_CORE)/libdmlc.a:
	+	cd $(DMLC_CORE); make libdmlc.a; cd -

build/src/%.o: src/%.cc
	@mkdir -p $(@D)
	$(CXX) $(PKG_CFLAGS) -MM -MT build/$*.o $< >build/src/$*.d
	$(CXX) -c $(PKG_CFLAGS) -c $< -o $@

${TARGET}: $(ALL_DEP)
	@mkdir -p $(@D)
	$(CXX) $(PKG_CFLAGS) -shared -o $@ $(filter %.o, $^) $(PKG_LDFLAGS) \
	-Wl,${WHOLE_ARCH} $(filter %.a, $^) -Wl,${NO_WHOLE_ARCH} $(PKG_LDFLAGS)

clean:
	$(RM) -rf build lib bin *~ */*~ */*/*~ */*/*/*~ */*.o */*/*.o */*/*/*.o */*.d */*/*.d */*/*/*.d

-include build/*.d
-include build/*/*.d
