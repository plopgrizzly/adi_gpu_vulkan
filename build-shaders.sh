# Aldaron's Device Interface / GPU                                             #
# Copyright (c) 2017 Plop Grizzly, Jeron Lau <jeron.lau@plopgrizzly.com>       #
# Licensed under the MIT LICENSE                                               #
#                             _        _                                       #
#                            /@\——————/@\                                      #
# .———  .                   |   o    o   |     .————      .            .       #
# |   | |  .———   .———   .——.     []     .——.  |      .——    ———: ———: | .   . #
# |   | |  |   |  |   |  \   \   <##>   /   /  |      |   |    /    /  | |   | #
# |———  |  |   |  |   |   |   ¯        ¯   |   |   -- |   |   /    /   |  \ /  #
# |     |  |   |  |———    |                |   |    | |   |  /    /    |   |   #
# |     |   ———   |       |                |    ————  |   | :——— :———  |   |   #
#                 |        \              /                              __/   #
#                           ¯————————————¯                                     #
# gen-spirv.sh                                                                 #

SPIRV_OPT="spirv-opt --strip-debug --freeze-spec-const --eliminate-dead-const --fold-spec-const-op-composite --unify-const"
SRC=src/shaders/glsl

OUT_UNOPTIMIZED=target/spv/unoptimized
OUT_OPTIMIZED=target/spv/optimized
OUT_RELEASE=target/spv/release

mkdir -p $OUT_UNOPTIMIZED/
mkdir -p $OUT_OPTIMIZED/
mkdir -p $OUT_RELEASE/

glslangValidator $SRC/solid-frag.glsl -V -o $OUT_UNOPTIMIZED/solid-frag.spv -S frag
glslangValidator $SRC/solid-vert.glsl -V -o $OUT_UNOPTIMIZED/solid-vert.spv -S vert
$SPIRV_OPT $OUT_UNOPTIMIZED/solid-frag.spv -o $OUT_OPTIMIZED/solid-frag.spv
$SPIRV_OPT $OUT_UNOPTIMIZED/solid-vert.spv -o $OUT_OPTIMIZED/solid-vert.spv

glslangValidator $SRC/gradient-frag.glsl -V -o $OUT_UNOPTIMIZED/gradient-frag.spv -S frag
glslangValidator $SRC/gradient-vert.glsl -V -o $OUT_UNOPTIMIZED/gradient-vert.spv -S vert
$SPIRV_OPT $OUT_UNOPTIMIZED/gradient-frag.spv -o $OUT_OPTIMIZED/gradient-frag.spv
$SPIRV_OPT $OUT_UNOPTIMIZED/gradient-vert.spv -o $OUT_OPTIMIZED/gradient-vert.spv

glslangValidator $SRC/texture-frag.glsl -V -o $OUT_UNOPTIMIZED/texture-frag.spv -S frag
glslangValidator $SRC/texture-vert.glsl -V -o $OUT_UNOPTIMIZED/texture-vert.spv -S vert
$SPIRV_OPT $OUT_UNOPTIMIZED/texture-frag.spv -o $OUT_OPTIMIZED/texture-frag.spv
$SPIRV_OPT $OUT_UNOPTIMIZED/texture-vert.spv -o $OUT_OPTIMIZED/texture-vert.spv

glslangValidator $SRC/faded-frag.glsl -V -o $OUT_UNOPTIMIZED/faded-frag.spv -S frag
glslangValidator $SRC/faded-vert.glsl -V -o $OUT_UNOPTIMIZED/faded-vert.spv -S vert
$SPIRV_OPT $OUT_UNOPTIMIZED/faded-frag.spv -o $OUT_OPTIMIZED/faded-frag.spv
$SPIRV_OPT $OUT_UNOPTIMIZED/faded-vert.spv -o $OUT_OPTIMIZED/faded-vert.spv

glslangValidator $SRC/tinted-frag.glsl -V -o $OUT_UNOPTIMIZED/tinted-frag.spv -S frag
glslangValidator $SRC/tinted-vert.glsl -V -o $OUT_UNOPTIMIZED/tinted-vert.spv -S vert
$SPIRV_OPT $OUT_UNOPTIMIZED/tinted-frag.spv -o $OUT_OPTIMIZED/tinted-frag.spv
$SPIRV_OPT $OUT_UNOPTIMIZED/tinted-vert.spv -o $OUT_OPTIMIZED/tinted-vert.spv

glslangValidator $SRC/complex-frag.glsl -V -o $OUT_UNOPTIMIZED/complex-frag.spv -S frag
glslangValidator $SRC/complex-vert.glsl -V -o $OUT_UNOPTIMIZED/complex-vert.spv -S vert
$SPIRV_OPT $OUT_UNOPTIMIZED/complex-frag.spv -o $OUT_OPTIMIZED/complex-frag.spv
$SPIRV_OPT $OUT_UNOPTIMIZED/complex-vert.spv -o $OUT_OPTIMIZED/complex-vert.spv

spirv-remap --map all --dce all --strip-all --input $OUT_OPTIMIZED/*.spv --output $OUT_RELEASE/

cp $OUT_RELEASE/* src/shaders/res/
# cp $OUT_UNOPTIMIZED/* src/native_renderer/vulkan/res/
