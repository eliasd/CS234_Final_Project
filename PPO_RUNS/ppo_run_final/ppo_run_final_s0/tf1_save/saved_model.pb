ак
є'▀&
:
Add
x"T
y"T
z"T"
Ttype:
2	
A
AddV2
x"T
y"T
z"T"
Ttype:
2	ђљ
Ь
	ApplyAdam
var"Tђ	
m"Tђ	
v"Tђ
beta1_power"T
beta2_power"T
lr"T

beta1"T

beta2"T
epsilon"T	
grad"T
out"Tђ" 
Ttype:
2	"
use_lockingbool( "
use_nesterovbool( 
x
Assign
ref"Tђ

value"T

output_ref"Tђ"	
Ttype"
validate_shapebool("
use_lockingbool(ў
~
BiasAdd

value"T	
bias"T
output"T" 
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
~
BiasAddGrad
out_backprop"T
output"T" 
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
R
BroadcastGradientArgs
s0"T
s1"T
r0"T
r1"T"
Ttype0:
2	
N
Cast	
x"SrcT	
y"DstT"
SrcTtype"
DstTtype"
Truncatebool( 
h
ConcatV2
values"T*N
axis"Tidx
output"T"
Nint(0"	
Ttype"
Tidxtype0:
2	
8
Const
output"dtype"
valuetensor"
dtypetype
S
DynamicStitch
indices*N
data"T*N
merged"T"
Nint(0"	
Ttype
,
Exp
x"T
y"T"
Ttype:

2
^
Fill
dims"
index_type

value"T
output"T"	
Ttype"

index_typetype0:
2	
?
FloorDiv
x"T
y"T
z"T"
Ttype:
2	
9
FloorMod
x"T
y"T
z"T"
Ttype:

2	
=
Greater
x"T
y"T
z
"
Ttype:
2	
.
Identity

input"T
output"T"	
Ttype
:
Less
x"T
y"T
z
"
Ttype:
2	
?
	LessEqual
x"T
y"T
z
"
Ttype:
2	
,
Log
x"T
y"T"
Ttype:

2
?

LogSoftmax
logits"T

logsoftmax"T"
Ttype:
2
#
	LogicalOr
x

y

z
љ
q
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:

2	
8
Maximum
x"T
y"T
z"T"
Ttype:

2	
Ї
Mean

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( " 
Ttype:
2	"
Tidxtype0:
2	
e
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool(ѕ
8
Minimum
x"T
y"T
z"T"
Ttype:

2	
=
Mul
x"T
y"T
z"T"
Ttype:
2	љ
е
Multinomial
logits"T
num_samples
output"output_dtype"
seedint "
seed2int "
Ttype:
2	" 
output_dtypetype0	:
2	ѕ
.
Neg
x"T
y"T"
Ttype:

2	

NoOp
ї
OneHot
indices"TI	
depth
on_value"T
	off_value"T
output"T"
axisint         "	
Ttype"
TItype0	:
2	
M
Pack
values"T*N
output"T"
Nint(0"	
Ttype"
axisint 
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
X
PlaceholderWithDefault
input"dtype
output"dtype"
dtypetype"
shapeshape
6
Pow
x"T
y"T
z"T"
Ttype:

2	
Ї
Prod

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( " 
Ttype:
2	"
Tidxtype0:
2	
e
PyFunc
input2Tin
output2Tout"
tokenstring"
Tin
list(type)("
Tout
list(type)(ѕ
~
RandomUniform

shape"T
output"dtype"
seedint "
seed2int "
dtypetype:
2"
Ttype:
2	ѕ
a
Range
start"Tidx
limit"Tidx
delta"Tidx
output"Tidx"
Tidxtype0:	
2	
>
RealDiv
x"T
y"T
z"T"
Ttype:
2	
[
Reshape
tensor"T
shape"Tshape
output"T"	
Ttype"
Tshapetype0:
2	
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0ѕ
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0ѕ
?
Select
	condition

t"T
e"T
output"T"	
Ttype
P
Shape

input"T
output"out_type"	
Ttype"
out_typetype0:
2	
H
ShardedFilename
basename	
shard

num_shards
filename
І
SplitV

value"T
size_splits"Tlen
	split_dim
output"T*	num_split"
	num_splitint(0"	
Ttype"
Tlentype0	:
2	
N
Squeeze

input"T
output"T"	
Ttype"
squeeze_dims	list(int)
 (
N

StringJoin
inputs*N

output"
Nint(0"
	separatorstring 
:
Sub
x"T
y"T
z"T"
Ttype:
2	
ї
Sum

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( " 
Ttype:
2	"
Tidxtype0:
2	
-
Tanh
x"T
y"T"
Ttype:

2
:
TanhGrad
y"T
dy"T
z"T"
Ttype:

2
c
Tile

input"T
	multiples"
Tmultiples
output"T"	
Ttype"

Tmultiplestype0:
2	
s

VariableV2
ref"dtypeђ"
shapeshape"
dtypetype"
	containerstring "
shared_namestring ѕ
&
	ZerosLike
x"T
y"T"	
Ttype"serve*1.15.22v1.15.0-92-g5d80e1e┐В
p
PlaceholderPlaceholder*
dtype0*(
_output_shapes
:         Є*
shape:         Є
h
Placeholder_1Placeholder*
shape:         *
dtype0*#
_output_shapes
:         
h
Placeholder_2Placeholder*
dtype0*#
_output_shapes
:         *
shape:         
h
Placeholder_3Placeholder*
dtype0*#
_output_shapes
:         *
shape:         
h
Placeholder_4Placeholder*
dtype0*#
_output_shapes
:         *
shape:         
Ц
0pi/dense/kernel/Initializer/random_uniform/shapeConst*
valueB"Є   @   *"
_class
loc:@pi/dense/kernel*
dtype0*
_output_shapes
:
Ќ
.pi/dense/kernel/Initializer/random_uniform/minConst*
valueB
 *Ю╬1Й*"
_class
loc:@pi/dense/kernel*
dtype0*
_output_shapes
: 
Ќ
.pi/dense/kernel/Initializer/random_uniform/maxConst*
valueB
 *Ю╬1>*"
_class
loc:@pi/dense/kernel*
dtype0*
_output_shapes
: 
№
8pi/dense/kernel/Initializer/random_uniform/RandomUniformRandomUniform0pi/dense/kernel/Initializer/random_uniform/shape*
T0*"
_class
loc:@pi/dense/kernel*
seed2*
dtype0*
_output_shapes
:	Є@*

seed 
┌
.pi/dense/kernel/Initializer/random_uniform/subSub.pi/dense/kernel/Initializer/random_uniform/max.pi/dense/kernel/Initializer/random_uniform/min*
T0*"
_class
loc:@pi/dense/kernel*
_output_shapes
: 
ь
.pi/dense/kernel/Initializer/random_uniform/mulMul8pi/dense/kernel/Initializer/random_uniform/RandomUniform.pi/dense/kernel/Initializer/random_uniform/sub*
T0*"
_class
loc:@pi/dense/kernel*
_output_shapes
:	Є@
▀
*pi/dense/kernel/Initializer/random_uniformAdd.pi/dense/kernel/Initializer/random_uniform/mul.pi/dense/kernel/Initializer/random_uniform/min*
_output_shapes
:	Є@*
T0*"
_class
loc:@pi/dense/kernel
Е
pi/dense/kernel
VariableV2*"
_class
loc:@pi/dense/kernel*
	container *
shape:	Є@*
dtype0*
_output_shapes
:	Є@*
shared_name 
н
pi/dense/kernel/AssignAssignpi/dense/kernel*pi/dense/kernel/Initializer/random_uniform*"
_class
loc:@pi/dense/kernel*
validate_shape(*
_output_shapes
:	Є@*
use_locking(*
T0

pi/dense/kernel/readIdentitypi/dense/kernel*
T0*"
_class
loc:@pi/dense/kernel*
_output_shapes
:	Є@
ј
pi/dense/bias/Initializer/zerosConst*
valueB@*    * 
_class
loc:@pi/dense/bias*
dtype0*
_output_shapes
:@
Џ
pi/dense/bias
VariableV2*
	container *
shape:@*
dtype0*
_output_shapes
:@*
shared_name * 
_class
loc:@pi/dense/bias
Й
pi/dense/bias/AssignAssignpi/dense/biaspi/dense/bias/Initializer/zeros*
validate_shape(*
_output_shapes
:@*
use_locking(*
T0* 
_class
loc:@pi/dense/bias
t
pi/dense/bias/readIdentitypi/dense/bias*
_output_shapes
:@*
T0* 
_class
loc:@pi/dense/bias
ћ
pi/dense/MatMulMatMulPlaceholderpi/dense/kernel/read*
T0*'
_output_shapes
:         @*
transpose_a( *
transpose_b( 
Ѕ
pi/dense/BiasAddBiasAddpi/dense/MatMulpi/dense/bias/read*'
_output_shapes
:         @*
T0*
data_formatNHWC
Y
pi/dense/TanhTanhpi/dense/BiasAdd*'
_output_shapes
:         @*
T0
Е
2pi/dense_1/kernel/Initializer/random_uniform/shapeConst*
valueB"@   @   *$
_class
loc:@pi/dense_1/kernel*
dtype0*
_output_shapes
:
Џ
0pi/dense_1/kernel/Initializer/random_uniform/minConst*
dtype0*
_output_shapes
: *
valueB
 *О│]Й*$
_class
loc:@pi/dense_1/kernel
Џ
0pi/dense_1/kernel/Initializer/random_uniform/maxConst*
valueB
 *О│]>*$
_class
loc:@pi/dense_1/kernel*
dtype0*
_output_shapes
: 
З
:pi/dense_1/kernel/Initializer/random_uniform/RandomUniformRandomUniform2pi/dense_1/kernel/Initializer/random_uniform/shape*
dtype0*
_output_shapes

:@@*

seed *
T0*$
_class
loc:@pi/dense_1/kernel*
seed2
Р
0pi/dense_1/kernel/Initializer/random_uniform/subSub0pi/dense_1/kernel/Initializer/random_uniform/max0pi/dense_1/kernel/Initializer/random_uniform/min*$
_class
loc:@pi/dense_1/kernel*
_output_shapes
: *
T0
З
0pi/dense_1/kernel/Initializer/random_uniform/mulMul:pi/dense_1/kernel/Initializer/random_uniform/RandomUniform0pi/dense_1/kernel/Initializer/random_uniform/sub*
_output_shapes

:@@*
T0*$
_class
loc:@pi/dense_1/kernel
Т
,pi/dense_1/kernel/Initializer/random_uniformAdd0pi/dense_1/kernel/Initializer/random_uniform/mul0pi/dense_1/kernel/Initializer/random_uniform/min*
T0*$
_class
loc:@pi/dense_1/kernel*
_output_shapes

:@@
Ф
pi/dense_1/kernel
VariableV2*
shared_name *$
_class
loc:@pi/dense_1/kernel*
	container *
shape
:@@*
dtype0*
_output_shapes

:@@
█
pi/dense_1/kernel/AssignAssignpi/dense_1/kernel,pi/dense_1/kernel/Initializer/random_uniform*
use_locking(*
T0*$
_class
loc:@pi/dense_1/kernel*
validate_shape(*
_output_shapes

:@@
ё
pi/dense_1/kernel/readIdentitypi/dense_1/kernel*
T0*$
_class
loc:@pi/dense_1/kernel*
_output_shapes

:@@
њ
!pi/dense_1/bias/Initializer/zerosConst*
valueB@*    *"
_class
loc:@pi/dense_1/bias*
dtype0*
_output_shapes
:@
Ъ
pi/dense_1/bias
VariableV2*
shared_name *"
_class
loc:@pi/dense_1/bias*
	container *
shape:@*
dtype0*
_output_shapes
:@
к
pi/dense_1/bias/AssignAssignpi/dense_1/bias!pi/dense_1/bias/Initializer/zeros*
_output_shapes
:@*
use_locking(*
T0*"
_class
loc:@pi/dense_1/bias*
validate_shape(
z
pi/dense_1/bias/readIdentitypi/dense_1/bias*
T0*"
_class
loc:@pi/dense_1/bias*
_output_shapes
:@
џ
pi/dense_1/MatMulMatMulpi/dense/Tanhpi/dense_1/kernel/read*
transpose_b( *
T0*'
_output_shapes
:         @*
transpose_a( 
Ј
pi/dense_1/BiasAddBiasAddpi/dense_1/MatMulpi/dense_1/bias/read*
T0*
data_formatNHWC*'
_output_shapes
:         @
]
pi/dense_1/TanhTanhpi/dense_1/BiasAdd*'
_output_shapes
:         @*
T0
Е
2pi/dense_2/kernel/Initializer/random_uniform/shapeConst*
valueB"@      *$
_class
loc:@pi/dense_2/kernel*
dtype0*
_output_shapes
:
Џ
0pi/dense_2/kernel/Initializer/random_uniform/minConst*
valueB
 *0ўЙ*$
_class
loc:@pi/dense_2/kernel*
dtype0*
_output_shapes
: 
Џ
0pi/dense_2/kernel/Initializer/random_uniform/maxConst*
dtype0*
_output_shapes
: *
valueB
 *0ў>*$
_class
loc:@pi/dense_2/kernel
З
:pi/dense_2/kernel/Initializer/random_uniform/RandomUniformRandomUniform2pi/dense_2/kernel/Initializer/random_uniform/shape*
dtype0*
_output_shapes

:@*

seed *
T0*$
_class
loc:@pi/dense_2/kernel*
seed2*
Р
0pi/dense_2/kernel/Initializer/random_uniform/subSub0pi/dense_2/kernel/Initializer/random_uniform/max0pi/dense_2/kernel/Initializer/random_uniform/min*
T0*$
_class
loc:@pi/dense_2/kernel*
_output_shapes
: 
З
0pi/dense_2/kernel/Initializer/random_uniform/mulMul:pi/dense_2/kernel/Initializer/random_uniform/RandomUniform0pi/dense_2/kernel/Initializer/random_uniform/sub*
T0*$
_class
loc:@pi/dense_2/kernel*
_output_shapes

:@
Т
,pi/dense_2/kernel/Initializer/random_uniformAdd0pi/dense_2/kernel/Initializer/random_uniform/mul0pi/dense_2/kernel/Initializer/random_uniform/min*
T0*$
_class
loc:@pi/dense_2/kernel*
_output_shapes

:@
Ф
pi/dense_2/kernel
VariableV2*
shared_name *$
_class
loc:@pi/dense_2/kernel*
	container *
shape
:@*
dtype0*
_output_shapes

:@
█
pi/dense_2/kernel/AssignAssignpi/dense_2/kernel,pi/dense_2/kernel/Initializer/random_uniform*
_output_shapes

:@*
use_locking(*
T0*$
_class
loc:@pi/dense_2/kernel*
validate_shape(
ё
pi/dense_2/kernel/readIdentitypi/dense_2/kernel*
_output_shapes

:@*
T0*$
_class
loc:@pi/dense_2/kernel
њ
!pi/dense_2/bias/Initializer/zerosConst*
valueB*    *"
_class
loc:@pi/dense_2/bias*
dtype0*
_output_shapes
:
Ъ
pi/dense_2/bias
VariableV2*
_output_shapes
:*
shared_name *"
_class
loc:@pi/dense_2/bias*
	container *
shape:*
dtype0
к
pi/dense_2/bias/AssignAssignpi/dense_2/bias!pi/dense_2/bias/Initializer/zeros*
use_locking(*
T0*"
_class
loc:@pi/dense_2/bias*
validate_shape(*
_output_shapes
:
z
pi/dense_2/bias/readIdentitypi/dense_2/bias*"
_class
loc:@pi/dense_2/bias*
_output_shapes
:*
T0
ю
pi/dense_2/MatMulMatMulpi/dense_1/Tanhpi/dense_2/kernel/read*'
_output_shapes
:         *
transpose_a( *
transpose_b( *
T0
Ј
pi/dense_2/BiasAddBiasAddpi/dense_2/MatMulpi/dense_2/bias/read*'
_output_shapes
:         *
T0*
data_formatNHWC
a
pi/LogSoftmax
LogSoftmaxpi/dense_2/BiasAdd*
T0*'
_output_shapes
:         
h
&pi/multinomial/Multinomial/num_samplesConst*
value	B :*
dtype0*
_output_shapes
: 
─
pi/multinomial/MultinomialMultinomialpi/dense_2/BiasAdd&pi/multinomial/Multinomial/num_samples*
T0*'
_output_shapes
:         *
seed28*

seed *
output_dtype0	
v

pi/SqueezeSqueezepi/multinomial/Multinomial*
squeeze_dims
*
T0	*#
_output_shapes
:         
X
pi/one_hot/on_valueConst*
_output_shapes
: *
valueB
 *  ђ?*
dtype0
Y
pi/one_hot/off_valueConst*
_output_shapes
: *
valueB
 *    *
dtype0
R
pi/one_hot/depthConst*
value	B :*
dtype0*
_output_shapes
: 
▒

pi/one_hotOneHotPlaceholder_1pi/one_hot/depthpi/one_hot/on_valuepi/one_hot/off_value*
T0*
axis         *
TI0*'
_output_shapes
:         
Z
pi/mulMul
pi/one_hotpi/LogSoftmax*
T0*'
_output_shapes
:         
Z
pi/Sum/reduction_indicesConst*
value	B :*
dtype0*
_output_shapes
: 
z
pi/SumSumpi/mulpi/Sum/reduction_indices*
	keep_dims( *

Tidx0*
T0*#
_output_shapes
:         
Z
pi/one_hot_1/on_valueConst*
valueB
 *  ђ?*
dtype0*
_output_shapes
: 
[
pi/one_hot_1/off_valueConst*
valueB
 *    *
dtype0*
_output_shapes
: 
T
pi/one_hot_1/depthConst*
value	B :*
dtype0*
_output_shapes
: 
Х
pi/one_hot_1OneHot
pi/Squeezepi/one_hot_1/depthpi/one_hot_1/on_valuepi/one_hot_1/off_value*
T0*
axis         *
TI0	*'
_output_shapes
:         
^
pi/mul_1Mulpi/one_hot_1pi/LogSoftmax*
T0*'
_output_shapes
:         
\
pi/Sum_1/reduction_indicesConst*
value	B :*
dtype0*
_output_shapes
: 
ђ
pi/Sum_1Sumpi/mul_1pi/Sum_1/reduction_indices*#
_output_shapes
:         *
	keep_dims( *

Tidx0*
T0
Б
/v/dense/kernel/Initializer/random_uniform/shapeConst*
_output_shapes
:*
valueB"Є   @   *!
_class
loc:@v/dense/kernel*
dtype0
Ћ
-v/dense/kernel/Initializer/random_uniform/minConst*
valueB
 *Ю╬1Й*!
_class
loc:@v/dense/kernel*
dtype0*
_output_shapes
: 
Ћ
-v/dense/kernel/Initializer/random_uniform/maxConst*
valueB
 *Ю╬1>*!
_class
loc:@v/dense/kernel*
dtype0*
_output_shapes
: 
В
7v/dense/kernel/Initializer/random_uniform/RandomUniformRandomUniform/v/dense/kernel/Initializer/random_uniform/shape*
dtype0*
_output_shapes
:	Є@*

seed *
T0*!
_class
loc:@v/dense/kernel*
seed2L
о
-v/dense/kernel/Initializer/random_uniform/subSub-v/dense/kernel/Initializer/random_uniform/max-v/dense/kernel/Initializer/random_uniform/min*
T0*!
_class
loc:@v/dense/kernel*
_output_shapes
: 
ж
-v/dense/kernel/Initializer/random_uniform/mulMul7v/dense/kernel/Initializer/random_uniform/RandomUniform-v/dense/kernel/Initializer/random_uniform/sub*
_output_shapes
:	Є@*
T0*!
_class
loc:@v/dense/kernel
█
)v/dense/kernel/Initializer/random_uniformAdd-v/dense/kernel/Initializer/random_uniform/mul-v/dense/kernel/Initializer/random_uniform/min*
T0*!
_class
loc:@v/dense/kernel*
_output_shapes
:	Є@
Д
v/dense/kernel
VariableV2*
shared_name *!
_class
loc:@v/dense/kernel*
	container *
shape:	Є@*
dtype0*
_output_shapes
:	Є@
л
v/dense/kernel/AssignAssignv/dense/kernel)v/dense/kernel/Initializer/random_uniform*
use_locking(*
T0*!
_class
loc:@v/dense/kernel*
validate_shape(*
_output_shapes
:	Є@
|
v/dense/kernel/readIdentityv/dense/kernel*
T0*!
_class
loc:@v/dense/kernel*
_output_shapes
:	Є@
ї
v/dense/bias/Initializer/zerosConst*
valueB@*    *
_class
loc:@v/dense/bias*
dtype0*
_output_shapes
:@
Ў
v/dense/bias
VariableV2*
shared_name *
_class
loc:@v/dense/bias*
	container *
shape:@*
dtype0*
_output_shapes
:@
║
v/dense/bias/AssignAssignv/dense/biasv/dense/bias/Initializer/zeros*
T0*
_class
loc:@v/dense/bias*
validate_shape(*
_output_shapes
:@*
use_locking(
q
v/dense/bias/readIdentityv/dense/bias*
T0*
_class
loc:@v/dense/bias*
_output_shapes
:@
њ
v/dense/MatMulMatMulPlaceholderv/dense/kernel/read*
T0*'
_output_shapes
:         @*
transpose_a( *
transpose_b( 
є
v/dense/BiasAddBiasAddv/dense/MatMulv/dense/bias/read*
T0*
data_formatNHWC*'
_output_shapes
:         @
W
v/dense/TanhTanhv/dense/BiasAdd*
T0*'
_output_shapes
:         @
Д
1v/dense_1/kernel/Initializer/random_uniform/shapeConst*
valueB"@   @   *#
_class
loc:@v/dense_1/kernel*
dtype0*
_output_shapes
:
Ў
/v/dense_1/kernel/Initializer/random_uniform/minConst*
valueB
 *О│]Й*#
_class
loc:@v/dense_1/kernel*
dtype0*
_output_shapes
: 
Ў
/v/dense_1/kernel/Initializer/random_uniform/maxConst*
_output_shapes
: *
valueB
 *О│]>*#
_class
loc:@v/dense_1/kernel*
dtype0
ы
9v/dense_1/kernel/Initializer/random_uniform/RandomUniformRandomUniform1v/dense_1/kernel/Initializer/random_uniform/shape*
T0*#
_class
loc:@v/dense_1/kernel*
seed2]*
dtype0*
_output_shapes

:@@*

seed 
я
/v/dense_1/kernel/Initializer/random_uniform/subSub/v/dense_1/kernel/Initializer/random_uniform/max/v/dense_1/kernel/Initializer/random_uniform/min*
T0*#
_class
loc:@v/dense_1/kernel*
_output_shapes
: 
­
/v/dense_1/kernel/Initializer/random_uniform/mulMul9v/dense_1/kernel/Initializer/random_uniform/RandomUniform/v/dense_1/kernel/Initializer/random_uniform/sub*
T0*#
_class
loc:@v/dense_1/kernel*
_output_shapes

:@@
Р
+v/dense_1/kernel/Initializer/random_uniformAdd/v/dense_1/kernel/Initializer/random_uniform/mul/v/dense_1/kernel/Initializer/random_uniform/min*
T0*#
_class
loc:@v/dense_1/kernel*
_output_shapes

:@@
Е
v/dense_1/kernel
VariableV2*
	container *
shape
:@@*
dtype0*
_output_shapes

:@@*
shared_name *#
_class
loc:@v/dense_1/kernel
О
v/dense_1/kernel/AssignAssignv/dense_1/kernel+v/dense_1/kernel/Initializer/random_uniform*
T0*#
_class
loc:@v/dense_1/kernel*
validate_shape(*
_output_shapes

:@@*
use_locking(
Ђ
v/dense_1/kernel/readIdentityv/dense_1/kernel*
T0*#
_class
loc:@v/dense_1/kernel*
_output_shapes

:@@
љ
 v/dense_1/bias/Initializer/zerosConst*
_output_shapes
:@*
valueB@*    *!
_class
loc:@v/dense_1/bias*
dtype0
Ю
v/dense_1/bias
VariableV2*
dtype0*
_output_shapes
:@*
shared_name *!
_class
loc:@v/dense_1/bias*
	container *
shape:@
┬
v/dense_1/bias/AssignAssignv/dense_1/bias v/dense_1/bias/Initializer/zeros*
use_locking(*
T0*!
_class
loc:@v/dense_1/bias*
validate_shape(*
_output_shapes
:@
w
v/dense_1/bias/readIdentityv/dense_1/bias*
T0*!
_class
loc:@v/dense_1/bias*
_output_shapes
:@
Ќ
v/dense_1/MatMulMatMulv/dense/Tanhv/dense_1/kernel/read*'
_output_shapes
:         @*
transpose_a( *
transpose_b( *
T0
ї
v/dense_1/BiasAddBiasAddv/dense_1/MatMulv/dense_1/bias/read*
T0*
data_formatNHWC*'
_output_shapes
:         @
[
v/dense_1/TanhTanhv/dense_1/BiasAdd*'
_output_shapes
:         @*
T0
Д
1v/dense_2/kernel/Initializer/random_uniform/shapeConst*
valueB"@      *#
_class
loc:@v/dense_2/kernel*
dtype0*
_output_shapes
:
Ў
/v/dense_2/kernel/Initializer/random_uniform/minConst*
dtype0*
_output_shapes
: *
valueB
 *ѕјЏЙ*#
_class
loc:@v/dense_2/kernel
Ў
/v/dense_2/kernel/Initializer/random_uniform/maxConst*
valueB
 *ѕјЏ>*#
_class
loc:@v/dense_2/kernel*
dtype0*
_output_shapes
: 
ы
9v/dense_2/kernel/Initializer/random_uniform/RandomUniformRandomUniform1v/dense_2/kernel/Initializer/random_uniform/shape*

seed *
T0*#
_class
loc:@v/dense_2/kernel*
seed2n*
dtype0*
_output_shapes

:@
я
/v/dense_2/kernel/Initializer/random_uniform/subSub/v/dense_2/kernel/Initializer/random_uniform/max/v/dense_2/kernel/Initializer/random_uniform/min*
T0*#
_class
loc:@v/dense_2/kernel*
_output_shapes
: 
­
/v/dense_2/kernel/Initializer/random_uniform/mulMul9v/dense_2/kernel/Initializer/random_uniform/RandomUniform/v/dense_2/kernel/Initializer/random_uniform/sub*
T0*#
_class
loc:@v/dense_2/kernel*
_output_shapes

:@
Р
+v/dense_2/kernel/Initializer/random_uniformAdd/v/dense_2/kernel/Initializer/random_uniform/mul/v/dense_2/kernel/Initializer/random_uniform/min*#
_class
loc:@v/dense_2/kernel*
_output_shapes

:@*
T0
Е
v/dense_2/kernel
VariableV2*
dtype0*
_output_shapes

:@*
shared_name *#
_class
loc:@v/dense_2/kernel*
	container *
shape
:@
О
v/dense_2/kernel/AssignAssignv/dense_2/kernel+v/dense_2/kernel/Initializer/random_uniform*
validate_shape(*
_output_shapes

:@*
use_locking(*
T0*#
_class
loc:@v/dense_2/kernel
Ђ
v/dense_2/kernel/readIdentityv/dense_2/kernel*
T0*#
_class
loc:@v/dense_2/kernel*
_output_shapes

:@
љ
 v/dense_2/bias/Initializer/zerosConst*
valueB*    *!
_class
loc:@v/dense_2/bias*
dtype0*
_output_shapes
:
Ю
v/dense_2/bias
VariableV2*
dtype0*
_output_shapes
:*
shared_name *!
_class
loc:@v/dense_2/bias*
	container *
shape:
┬
v/dense_2/bias/AssignAssignv/dense_2/bias v/dense_2/bias/Initializer/zeros*!
_class
loc:@v/dense_2/bias*
validate_shape(*
_output_shapes
:*
use_locking(*
T0
w
v/dense_2/bias/readIdentityv/dense_2/bias*
T0*!
_class
loc:@v/dense_2/bias*
_output_shapes
:
Ў
v/dense_2/MatMulMatMulv/dense_1/Tanhv/dense_2/kernel/read*'
_output_shapes
:         *
transpose_a( *
transpose_b( *
T0
ї
v/dense_2/BiasAddBiasAddv/dense_2/MatMulv/dense_2/bias/read*
T0*
data_formatNHWC*'
_output_shapes
:         
l
	v/SqueezeSqueezev/dense_2/BiasAdd*
squeeze_dims
*
T0*#
_output_shapes
:         
O
subSubpi/SumPlaceholder_4*
T0*#
_output_shapes
:         
=
ExpExpsub*#
_output_shapes
:         *
T0
N
	Greater/yConst*
valueB
 *    *
dtype0*
_output_shapes
: 
Z
GreaterGreaterPlaceholder_2	Greater/y*#
_output_shapes
:         *
T0
J
mul/xConst*
valueB
 *џЎЎ?*
dtype0*
_output_shapes
: 
N
mulMulmul/xPlaceholder_2*#
_output_shapes
:         *
T0
L
mul_1/xConst*
valueB
 *═╠L?*
dtype0*
_output_shapes
: 
R
mul_1Mulmul_1/xPlaceholder_2*#
_output_shapes
:         *
T0
S
SelectSelectGreatermulmul_1*#
_output_shapes
:         *
T0
N
mul_2MulExpPlaceholder_2*
T0*#
_output_shapes
:         
O
MinimumMinimummul_2Select*#
_output_shapes
:         *
T0
O
ConstConst*
valueB: *
dtype0*
_output_shapes
:
Z
MeanMeanMinimumConst*
T0*
_output_shapes
: *
	keep_dims( *

Tidx0
1
NegNegMean*
T0*
_output_shapes
: 
T
sub_1SubPlaceholder_3	v/Squeeze*
T0*#
_output_shapes
:         
J
pow/yConst*
valueB
 *   @*
dtype0*
_output_shapes
: 
F
powPowsub_1pow/y*
T0*#
_output_shapes
:         
Q
Const_1Const*
valueB: *
dtype0*
_output_shapes
:
Z
Mean_1MeanpowConst_1*
	keep_dims( *

Tidx0*
T0*
_output_shapes
: 
Q
sub_2SubPlaceholder_4pi/Sum*
T0*#
_output_shapes
:         
Q
Const_2Const*
_output_shapes
:*
valueB: *
dtype0
\
Mean_2Meansub_2Const_2*
	keep_dims( *

Tidx0*
T0*
_output_shapes
: 
B
Neg_1Negpi/Sum*#
_output_shapes
:         *
T0
Q
Const_3Const*
_output_shapes
:*
valueB: *
dtype0
\
Mean_3MeanNeg_1Const_3*
T0*
_output_shapes
: *
	keep_dims( *

Tidx0
P
Greater_1/yConst*
valueB
 *џЎЎ?*
dtype0*
_output_shapes
: 
T
	Greater_1GreaterExpGreater_1/y*#
_output_shapes
:         *
T0
K
Less/yConst*
valueB
 *═╠L?*
dtype0*
_output_shapes
: 
G
LessLessExpLess/y*#
_output_shapes
:         *
T0
L
	LogicalOr	LogicalOr	Greater_1Less*#
_output_shapes
:         
d
CastCast	LogicalOr*#
_output_shapes
:         *

DstT0*

SrcT0
*
Truncate( 
Q
Const_4Const*
valueB: *
dtype0*
_output_shapes
:
[
Mean_4MeanCastConst_4*
_output_shapes
: *
	keep_dims( *

Tidx0*
T0
R
gradients/ShapeConst*
valueB *
dtype0*
_output_shapes
: 
X
gradients/grad_ys_0Const*
dtype0*
_output_shapes
: *
valueB
 *  ђ?
o
gradients/FillFillgradients/Shapegradients/grad_ys_0*
T0*

index_type0*
_output_shapes
: 
N
gradients/Neg_grad/NegNeggradients/Fill*
T0*
_output_shapes
: 
k
!gradients/Mean_grad/Reshape/shapeConst*
_output_shapes
:*
valueB:*
dtype0
ћ
gradients/Mean_grad/ReshapeReshapegradients/Neg_grad/Neg!gradients/Mean_grad/Reshape/shape*
T0*
Tshape0*
_output_shapes
:
`
gradients/Mean_grad/ShapeShapeMinimum*
_output_shapes
:*
T0*
out_type0
ў
gradients/Mean_grad/TileTilegradients/Mean_grad/Reshapegradients/Mean_grad/Shape*

Tmultiples0*
T0*#
_output_shapes
:         
b
gradients/Mean_grad/Shape_1ShapeMinimum*
T0*
out_type0*
_output_shapes
:
^
gradients/Mean_grad/Shape_2Const*
valueB *
dtype0*
_output_shapes
: 
c
gradients/Mean_grad/ConstConst*
valueB: *
dtype0*
_output_shapes
:
ќ
gradients/Mean_grad/ProdProdgradients/Mean_grad/Shape_1gradients/Mean_grad/Const*
T0*
_output_shapes
: *
	keep_dims( *

Tidx0
e
gradients/Mean_grad/Const_1Const*
dtype0*
_output_shapes
:*
valueB: 
џ
gradients/Mean_grad/Prod_1Prodgradients/Mean_grad/Shape_2gradients/Mean_grad/Const_1*
T0*
_output_shapes
: *
	keep_dims( *

Tidx0
_
gradients/Mean_grad/Maximum/yConst*
value	B :*
dtype0*
_output_shapes
: 
ѓ
gradients/Mean_grad/MaximumMaximumgradients/Mean_grad/Prod_1gradients/Mean_grad/Maximum/y*
T0*
_output_shapes
: 
ђ
gradients/Mean_grad/floordivFloorDivgradients/Mean_grad/Prodgradients/Mean_grad/Maximum*
_output_shapes
: *
T0
~
gradients/Mean_grad/CastCastgradients/Mean_grad/floordiv*

SrcT0*
Truncate( *
_output_shapes
: *

DstT0
ѕ
gradients/Mean_grad/truedivRealDivgradients/Mean_grad/Tilegradients/Mean_grad/Cast*#
_output_shapes
:         *
T0
a
gradients/Minimum_grad/ShapeShapemul_2*
T0*
out_type0*
_output_shapes
:
d
gradients/Minimum_grad/Shape_1ShapeSelect*
_output_shapes
:*
T0*
out_type0
y
gradients/Minimum_grad/Shape_2Shapegradients/Mean_grad/truediv*
T0*
out_type0*
_output_shapes
:
g
"gradients/Minimum_grad/zeros/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: 
е
gradients/Minimum_grad/zerosFillgradients/Minimum_grad/Shape_2"gradients/Minimum_grad/zeros/Const*#
_output_shapes
:         *
T0*

index_type0
j
 gradients/Minimum_grad/LessEqual	LessEqualmul_2Select*
T0*#
_output_shapes
:         
└
,gradients/Minimum_grad/BroadcastGradientArgsBroadcastGradientArgsgradients/Minimum_grad/Shapegradients/Minimum_grad/Shape_1*
T0*2
_output_shapes 
:         :         
▓
gradients/Minimum_grad/SelectSelect gradients/Minimum_grad/LessEqualgradients/Mean_grad/truedivgradients/Minimum_grad/zeros*
T0*#
_output_shapes
:         
«
gradients/Minimum_grad/SumSumgradients/Minimum_grad/Select,gradients/Minimum_grad/BroadcastGradientArgs*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0
Ъ
gradients/Minimum_grad/ReshapeReshapegradients/Minimum_grad/Sumgradients/Minimum_grad/Shape*
T0*
Tshape0*#
_output_shapes
:         
┤
gradients/Minimum_grad/Select_1Select gradients/Minimum_grad/LessEqualgradients/Minimum_grad/zerosgradients/Mean_grad/truediv*
T0*#
_output_shapes
:         
┤
gradients/Minimum_grad/Sum_1Sumgradients/Minimum_grad/Select_1.gradients/Minimum_grad/BroadcastGradientArgs:1*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0
Ц
 gradients/Minimum_grad/Reshape_1Reshapegradients/Minimum_grad/Sum_1gradients/Minimum_grad/Shape_1*
T0*
Tshape0*#
_output_shapes
:         
s
'gradients/Minimum_grad/tuple/group_depsNoOp^gradients/Minimum_grad/Reshape!^gradients/Minimum_grad/Reshape_1
Т
/gradients/Minimum_grad/tuple/control_dependencyIdentitygradients/Minimum_grad/Reshape(^gradients/Minimum_grad/tuple/group_deps*#
_output_shapes
:         *
T0*1
_class'
%#loc:@gradients/Minimum_grad/Reshape
В
1gradients/Minimum_grad/tuple/control_dependency_1Identity gradients/Minimum_grad/Reshape_1(^gradients/Minimum_grad/tuple/group_deps*3
_class)
'%loc:@gradients/Minimum_grad/Reshape_1*#
_output_shapes
:         *
T0
]
gradients/mul_2_grad/ShapeShapeExp*
out_type0*
_output_shapes
:*
T0
i
gradients/mul_2_grad/Shape_1ShapePlaceholder_2*
_output_shapes
:*
T0*
out_type0
║
*gradients/mul_2_grad/BroadcastGradientArgsBroadcastGradientArgsgradients/mul_2_grad/Shapegradients/mul_2_grad/Shape_1*
T0*2
_output_shapes 
:         :         
Ї
gradients/mul_2_grad/MulMul/gradients/Minimum_grad/tuple/control_dependencyPlaceholder_2*
T0*#
_output_shapes
:         
Ц
gradients/mul_2_grad/SumSumgradients/mul_2_grad/Mul*gradients/mul_2_grad/BroadcastGradientArgs*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
Ў
gradients/mul_2_grad/ReshapeReshapegradients/mul_2_grad/Sumgradients/mul_2_grad/Shape*
T0*
Tshape0*#
_output_shapes
:         
Ё
gradients/mul_2_grad/Mul_1MulExp/gradients/Minimum_grad/tuple/control_dependency*
T0*#
_output_shapes
:         
Ф
gradients/mul_2_grad/Sum_1Sumgradients/mul_2_grad/Mul_1,gradients/mul_2_grad/BroadcastGradientArgs:1*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
Ъ
gradients/mul_2_grad/Reshape_1Reshapegradients/mul_2_grad/Sum_1gradients/mul_2_grad/Shape_1*#
_output_shapes
:         *
T0*
Tshape0
m
%gradients/mul_2_grad/tuple/group_depsNoOp^gradients/mul_2_grad/Reshape^gradients/mul_2_grad/Reshape_1
я
-gradients/mul_2_grad/tuple/control_dependencyIdentitygradients/mul_2_grad/Reshape&^gradients/mul_2_grad/tuple/group_deps*
T0*/
_class%
#!loc:@gradients/mul_2_grad/Reshape*#
_output_shapes
:         
С
/gradients/mul_2_grad/tuple/control_dependency_1Identitygradients/mul_2_grad/Reshape_1&^gradients/mul_2_grad/tuple/group_deps*1
_class'
%#loc:@gradients/mul_2_grad/Reshape_1*#
_output_shapes
:         *
T0

gradients/Exp_grad/mulMul-gradients/mul_2_grad/tuple/control_dependencyExp*
T0*#
_output_shapes
:         
^
gradients/sub_grad/ShapeShapepi/Sum*
out_type0*
_output_shapes
:*
T0
g
gradients/sub_grad/Shape_1ShapePlaceholder_4*
_output_shapes
:*
T0*
out_type0
┤
(gradients/sub_grad/BroadcastGradientArgsBroadcastGradientArgsgradients/sub_grad/Shapegradients/sub_grad/Shape_1*2
_output_shapes 
:         :         *
T0
Ъ
gradients/sub_grad/SumSumgradients/Exp_grad/mul(gradients/sub_grad/BroadcastGradientArgs*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
Њ
gradients/sub_grad/ReshapeReshapegradients/sub_grad/Sumgradients/sub_grad/Shape*
T0*
Tshape0*#
_output_shapes
:         
c
gradients/sub_grad/NegNeggradients/Exp_grad/mul*
T0*#
_output_shapes
:         
Б
gradients/sub_grad/Sum_1Sumgradients/sub_grad/Neg*gradients/sub_grad/BroadcastGradientArgs:1*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
Ў
gradients/sub_grad/Reshape_1Reshapegradients/sub_grad/Sum_1gradients/sub_grad/Shape_1*
T0*
Tshape0*#
_output_shapes
:         
g
#gradients/sub_grad/tuple/group_depsNoOp^gradients/sub_grad/Reshape^gradients/sub_grad/Reshape_1
о
+gradients/sub_grad/tuple/control_dependencyIdentitygradients/sub_grad/Reshape$^gradients/sub_grad/tuple/group_deps*
T0*-
_class#
!loc:@gradients/sub_grad/Reshape*#
_output_shapes
:         
▄
-gradients/sub_grad/tuple/control_dependency_1Identitygradients/sub_grad/Reshape_1$^gradients/sub_grad/tuple/group_deps*#
_output_shapes
:         *
T0*/
_class%
#!loc:@gradients/sub_grad/Reshape_1
a
gradients/pi/Sum_grad/ShapeShapepi/mul*
T0*
out_type0*
_output_shapes
:
ї
gradients/pi/Sum_grad/SizeConst*
value	B :*.
_class$
" loc:@gradients/pi/Sum_grad/Shape*
dtype0*
_output_shapes
: 
Е
gradients/pi/Sum_grad/addAddV2pi/Sum/reduction_indicesgradients/pi/Sum_grad/Size*
T0*.
_class$
" loc:@gradients/pi/Sum_grad/Shape*
_output_shapes
: 
Г
gradients/pi/Sum_grad/modFloorModgradients/pi/Sum_grad/addgradients/pi/Sum_grad/Size*
T0*.
_class$
" loc:@gradients/pi/Sum_grad/Shape*
_output_shapes
: 
љ
gradients/pi/Sum_grad/Shape_1Const*
valueB *.
_class$
" loc:@gradients/pi/Sum_grad/Shape*
dtype0*
_output_shapes
: 
Њ
!gradients/pi/Sum_grad/range/startConst*
value	B : *.
_class$
" loc:@gradients/pi/Sum_grad/Shape*
dtype0*
_output_shapes
: 
Њ
!gradients/pi/Sum_grad/range/deltaConst*
value	B :*.
_class$
" loc:@gradients/pi/Sum_grad/Shape*
dtype0*
_output_shapes
: 
я
gradients/pi/Sum_grad/rangeRange!gradients/pi/Sum_grad/range/startgradients/pi/Sum_grad/Size!gradients/pi/Sum_grad/range/delta*
_output_shapes
:*

Tidx0*.
_class$
" loc:@gradients/pi/Sum_grad/Shape
њ
 gradients/pi/Sum_grad/Fill/valueConst*
value	B :*.
_class$
" loc:@gradients/pi/Sum_grad/Shape*
dtype0*
_output_shapes
: 
к
gradients/pi/Sum_grad/FillFillgradients/pi/Sum_grad/Shape_1 gradients/pi/Sum_grad/Fill/value*
_output_shapes
: *
T0*

index_type0*.
_class$
" loc:@gradients/pi/Sum_grad/Shape
Ѓ
#gradients/pi/Sum_grad/DynamicStitchDynamicStitchgradients/pi/Sum_grad/rangegradients/pi/Sum_grad/modgradients/pi/Sum_grad/Shapegradients/pi/Sum_grad/Fill*
T0*.
_class$
" loc:@gradients/pi/Sum_grad/Shape*
N*
_output_shapes
:
Љ
gradients/pi/Sum_grad/Maximum/yConst*
value	B :*.
_class$
" loc:@gradients/pi/Sum_grad/Shape*
dtype0*
_output_shapes
: 
├
gradients/pi/Sum_grad/MaximumMaximum#gradients/pi/Sum_grad/DynamicStitchgradients/pi/Sum_grad/Maximum/y*.
_class$
" loc:@gradients/pi/Sum_grad/Shape*
_output_shapes
:*
T0
╗
gradients/pi/Sum_grad/floordivFloorDivgradients/pi/Sum_grad/Shapegradients/pi/Sum_grad/Maximum*.
_class$
" loc:@gradients/pi/Sum_grad/Shape*
_output_shapes
:*
T0
├
gradients/pi/Sum_grad/ReshapeReshape+gradients/sub_grad/tuple/control_dependency#gradients/pi/Sum_grad/DynamicStitch*
T0*
Tshape0*0
_output_shapes
:                  
Ц
gradients/pi/Sum_grad/TileTilegradients/pi/Sum_grad/Reshapegradients/pi/Sum_grad/floordiv*'
_output_shapes
:         *

Tmultiples0*
T0
e
gradients/pi/mul_grad/ShapeShape
pi/one_hot*
_output_shapes
:*
T0*
out_type0
j
gradients/pi/mul_grad/Shape_1Shapepi/LogSoftmax*
T0*
out_type0*
_output_shapes
:
й
+gradients/pi/mul_grad/BroadcastGradientArgsBroadcastGradientArgsgradients/pi/mul_grad/Shapegradients/pi/mul_grad/Shape_1*2
_output_shapes 
:         :         *
T0
}
gradients/pi/mul_grad/MulMulgradients/pi/Sum_grad/Tilepi/LogSoftmax*
T0*'
_output_shapes
:         
е
gradients/pi/mul_grad/SumSumgradients/pi/mul_grad/Mul+gradients/pi/mul_grad/BroadcastGradientArgs*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
а
gradients/pi/mul_grad/ReshapeReshapegradients/pi/mul_grad/Sumgradients/pi/mul_grad/Shape*
T0*
Tshape0*'
_output_shapes
:         
|
gradients/pi/mul_grad/Mul_1Mul
pi/one_hotgradients/pi/Sum_grad/Tile*
T0*'
_output_shapes
:         
«
gradients/pi/mul_grad/Sum_1Sumgradients/pi/mul_grad/Mul_1-gradients/pi/mul_grad/BroadcastGradientArgs:1*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
д
gradients/pi/mul_grad/Reshape_1Reshapegradients/pi/mul_grad/Sum_1gradients/pi/mul_grad/Shape_1*
T0*
Tshape0*'
_output_shapes
:         
p
&gradients/pi/mul_grad/tuple/group_depsNoOp^gradients/pi/mul_grad/Reshape ^gradients/pi/mul_grad/Reshape_1
Т
.gradients/pi/mul_grad/tuple/control_dependencyIdentitygradients/pi/mul_grad/Reshape'^gradients/pi/mul_grad/tuple/group_deps*
T0*0
_class&
$"loc:@gradients/pi/mul_grad/Reshape*'
_output_shapes
:         
В
0gradients/pi/mul_grad/tuple/control_dependency_1Identitygradients/pi/mul_grad/Reshape_1'^gradients/pi/mul_grad/tuple/group_deps*
T0*2
_class(
&$loc:@gradients/pi/mul_grad/Reshape_1*'
_output_shapes
:         
h
 gradients/pi/LogSoftmax_grad/ExpExppi/LogSoftmax*
T0*'
_output_shapes
:         
}
2gradients/pi/LogSoftmax_grad/Sum/reduction_indicesConst*
valueB :
         *
dtype0*
_output_shapes
: 
▄
 gradients/pi/LogSoftmax_grad/SumSum0gradients/pi/mul_grad/tuple/control_dependency_12gradients/pi/LogSoftmax_grad/Sum/reduction_indices*
	keep_dims(*

Tidx0*
T0*'
_output_shapes
:         
Ю
 gradients/pi/LogSoftmax_grad/mulMul gradients/pi/LogSoftmax_grad/Sum gradients/pi/LogSoftmax_grad/Exp*
T0*'
_output_shapes
:         
Г
 gradients/pi/LogSoftmax_grad/subSub0gradients/pi/mul_grad/tuple/control_dependency_1 gradients/pi/LogSoftmax_grad/mul*'
_output_shapes
:         *
T0
џ
-gradients/pi/dense_2/BiasAdd_grad/BiasAddGradBiasAddGrad gradients/pi/LogSoftmax_grad/sub*
T0*
data_formatNHWC*
_output_shapes
:
Ї
2gradients/pi/dense_2/BiasAdd_grad/tuple/group_depsNoOp!^gradients/pi/LogSoftmax_grad/sub.^gradients/pi/dense_2/BiasAdd_grad/BiasAddGrad
ё
:gradients/pi/dense_2/BiasAdd_grad/tuple/control_dependencyIdentity gradients/pi/LogSoftmax_grad/sub3^gradients/pi/dense_2/BiasAdd_grad/tuple/group_deps*
T0*3
_class)
'%loc:@gradients/pi/LogSoftmax_grad/sub*'
_output_shapes
:         
Њ
<gradients/pi/dense_2/BiasAdd_grad/tuple/control_dependency_1Identity-gradients/pi/dense_2/BiasAdd_grad/BiasAddGrad3^gradients/pi/dense_2/BiasAdd_grad/tuple/group_deps*
_output_shapes
:*
T0*@
_class6
42loc:@gradients/pi/dense_2/BiasAdd_grad/BiasAddGrad
П
'gradients/pi/dense_2/MatMul_grad/MatMulMatMul:gradients/pi/dense_2/BiasAdd_grad/tuple/control_dependencypi/dense_2/kernel/read*
T0*'
_output_shapes
:         @*
transpose_a( *
transpose_b(
¤
)gradients/pi/dense_2/MatMul_grad/MatMul_1MatMulpi/dense_1/Tanh:gradients/pi/dense_2/BiasAdd_grad/tuple/control_dependency*
_output_shapes

:@*
transpose_a(*
transpose_b( *
T0
Ј
1gradients/pi/dense_2/MatMul_grad/tuple/group_depsNoOp(^gradients/pi/dense_2/MatMul_grad/MatMul*^gradients/pi/dense_2/MatMul_grad/MatMul_1
љ
9gradients/pi/dense_2/MatMul_grad/tuple/control_dependencyIdentity'gradients/pi/dense_2/MatMul_grad/MatMul2^gradients/pi/dense_2/MatMul_grad/tuple/group_deps*'
_output_shapes
:         @*
T0*:
_class0
.,loc:@gradients/pi/dense_2/MatMul_grad/MatMul
Ї
;gradients/pi/dense_2/MatMul_grad/tuple/control_dependency_1Identity)gradients/pi/dense_2/MatMul_grad/MatMul_12^gradients/pi/dense_2/MatMul_grad/tuple/group_deps*
T0*<
_class2
0.loc:@gradients/pi/dense_2/MatMul_grad/MatMul_1*
_output_shapes

:@
▒
'gradients/pi/dense_1/Tanh_grad/TanhGradTanhGradpi/dense_1/Tanh9gradients/pi/dense_2/MatMul_grad/tuple/control_dependency*'
_output_shapes
:         @*
T0
А
-gradients/pi/dense_1/BiasAdd_grad/BiasAddGradBiasAddGrad'gradients/pi/dense_1/Tanh_grad/TanhGrad*
T0*
data_formatNHWC*
_output_shapes
:@
ћ
2gradients/pi/dense_1/BiasAdd_grad/tuple/group_depsNoOp.^gradients/pi/dense_1/BiasAdd_grad/BiasAddGrad(^gradients/pi/dense_1/Tanh_grad/TanhGrad
њ
:gradients/pi/dense_1/BiasAdd_grad/tuple/control_dependencyIdentity'gradients/pi/dense_1/Tanh_grad/TanhGrad3^gradients/pi/dense_1/BiasAdd_grad/tuple/group_deps*
T0*:
_class0
.,loc:@gradients/pi/dense_1/Tanh_grad/TanhGrad*'
_output_shapes
:         @
Њ
<gradients/pi/dense_1/BiasAdd_grad/tuple/control_dependency_1Identity-gradients/pi/dense_1/BiasAdd_grad/BiasAddGrad3^gradients/pi/dense_1/BiasAdd_grad/tuple/group_deps*
T0*@
_class6
42loc:@gradients/pi/dense_1/BiasAdd_grad/BiasAddGrad*
_output_shapes
:@
П
'gradients/pi/dense_1/MatMul_grad/MatMulMatMul:gradients/pi/dense_1/BiasAdd_grad/tuple/control_dependencypi/dense_1/kernel/read*'
_output_shapes
:         @*
transpose_a( *
transpose_b(*
T0
═
)gradients/pi/dense_1/MatMul_grad/MatMul_1MatMulpi/dense/Tanh:gradients/pi/dense_1/BiasAdd_grad/tuple/control_dependency*
T0*
_output_shapes

:@@*
transpose_a(*
transpose_b( 
Ј
1gradients/pi/dense_1/MatMul_grad/tuple/group_depsNoOp(^gradients/pi/dense_1/MatMul_grad/MatMul*^gradients/pi/dense_1/MatMul_grad/MatMul_1
љ
9gradients/pi/dense_1/MatMul_grad/tuple/control_dependencyIdentity'gradients/pi/dense_1/MatMul_grad/MatMul2^gradients/pi/dense_1/MatMul_grad/tuple/group_deps*'
_output_shapes
:         @*
T0*:
_class0
.,loc:@gradients/pi/dense_1/MatMul_grad/MatMul
Ї
;gradients/pi/dense_1/MatMul_grad/tuple/control_dependency_1Identity)gradients/pi/dense_1/MatMul_grad/MatMul_12^gradients/pi/dense_1/MatMul_grad/tuple/group_deps*<
_class2
0.loc:@gradients/pi/dense_1/MatMul_grad/MatMul_1*
_output_shapes

:@@*
T0
Г
%gradients/pi/dense/Tanh_grad/TanhGradTanhGradpi/dense/Tanh9gradients/pi/dense_1/MatMul_grad/tuple/control_dependency*'
_output_shapes
:         @*
T0
Ю
+gradients/pi/dense/BiasAdd_grad/BiasAddGradBiasAddGrad%gradients/pi/dense/Tanh_grad/TanhGrad*
T0*
data_formatNHWC*
_output_shapes
:@
ј
0gradients/pi/dense/BiasAdd_grad/tuple/group_depsNoOp,^gradients/pi/dense/BiasAdd_grad/BiasAddGrad&^gradients/pi/dense/Tanh_grad/TanhGrad
і
8gradients/pi/dense/BiasAdd_grad/tuple/control_dependencyIdentity%gradients/pi/dense/Tanh_grad/TanhGrad1^gradients/pi/dense/BiasAdd_grad/tuple/group_deps*
T0*8
_class.
,*loc:@gradients/pi/dense/Tanh_grad/TanhGrad*'
_output_shapes
:         @
І
:gradients/pi/dense/BiasAdd_grad/tuple/control_dependency_1Identity+gradients/pi/dense/BiasAdd_grad/BiasAddGrad1^gradients/pi/dense/BiasAdd_grad/tuple/group_deps*
_output_shapes
:@*
T0*>
_class4
20loc:@gradients/pi/dense/BiasAdd_grad/BiasAddGrad
п
%gradients/pi/dense/MatMul_grad/MatMulMatMul8gradients/pi/dense/BiasAdd_grad/tuple/control_dependencypi/dense/kernel/read*(
_output_shapes
:         Є*
transpose_a( *
transpose_b(*
T0
╚
'gradients/pi/dense/MatMul_grad/MatMul_1MatMulPlaceholder8gradients/pi/dense/BiasAdd_grad/tuple/control_dependency*
T0*
_output_shapes
:	Є@*
transpose_a(*
transpose_b( 
Ѕ
/gradients/pi/dense/MatMul_grad/tuple/group_depsNoOp&^gradients/pi/dense/MatMul_grad/MatMul(^gradients/pi/dense/MatMul_grad/MatMul_1
Ѕ
7gradients/pi/dense/MatMul_grad/tuple/control_dependencyIdentity%gradients/pi/dense/MatMul_grad/MatMul0^gradients/pi/dense/MatMul_grad/tuple/group_deps*(
_output_shapes
:         Є*
T0*8
_class.
,*loc:@gradients/pi/dense/MatMul_grad/MatMul
є
9gradients/pi/dense/MatMul_grad/tuple/control_dependency_1Identity'gradients/pi/dense/MatMul_grad/MatMul_10^gradients/pi/dense/MatMul_grad/tuple/group_deps*
T0*:
_class0
.,loc:@gradients/pi/dense/MatMul_grad/MatMul_1*
_output_shapes
:	Є@
`
Reshape/shapeConst*
valueB:
         *
dtype0*
_output_shapes
:
љ
ReshapeReshape9gradients/pi/dense/MatMul_grad/tuple/control_dependency_1Reshape/shape*
_output_shapes	
:└C*
T0*
Tshape0
b
Reshape_1/shapeConst*
valueB:
         *
dtype0*
_output_shapes
:
ћ
	Reshape_1Reshape:gradients/pi/dense/BiasAdd_grad/tuple/control_dependency_1Reshape_1/shape*
_output_shapes
:@*
T0*
Tshape0
b
Reshape_2/shapeConst*
valueB:
         *
dtype0*
_output_shapes
:
ќ
	Reshape_2Reshape;gradients/pi/dense_1/MatMul_grad/tuple/control_dependency_1Reshape_2/shape*
Tshape0*
_output_shapes	
:ђ *
T0
b
Reshape_3/shapeConst*
valueB:
         *
dtype0*
_output_shapes
:
ќ
	Reshape_3Reshape<gradients/pi/dense_1/BiasAdd_grad/tuple/control_dependency_1Reshape_3/shape*
T0*
Tshape0*
_output_shapes
:@
b
Reshape_4/shapeConst*
valueB:
         *
dtype0*
_output_shapes
:
ќ
	Reshape_4Reshape;gradients/pi/dense_2/MatMul_grad/tuple/control_dependency_1Reshape_4/shape*
T0*
Tshape0*
_output_shapes	
:ђ
b
Reshape_5/shapeConst*
valueB:
         *
dtype0*
_output_shapes
:
ќ
	Reshape_5Reshape<gradients/pi/dense_2/BiasAdd_grad/tuple/control_dependency_1Reshape_5/shape*
T0*
Tshape0*
_output_shapes
:
M
concat/axisConst*
value	B : *
dtype0*
_output_shapes
: 
џ
concatConcatV2Reshape	Reshape_1	Reshape_2	Reshape_3	Reshape_4	Reshape_5concat/axis*
T0*
N*
_output_shapes	
:─f*

Tidx0
g
PyFuncPyFuncconcat*
_output_shapes	
:─f*
Tin
2*
Tout
2*
token
pyfunc_0
h
Const_5Const*
_output_shapes
:*-
value$B""└!  @      @         *
dtype0
Q
split/split_dimConst*
value	B : *
dtype0*
_output_shapes
: 
ћ
splitSplitVPyFuncConst_5split/split_dim*
T0*;
_output_shapes)
':└C:@:ђ :@:ђ:*
	num_split*

Tlen0
`
Reshape_6/shapeConst*
valueB"Є   @   *
dtype0*
_output_shapes
:
d
	Reshape_6ReshapesplitReshape_6/shape*
Tshape0*
_output_shapes
:	Є@*
T0
Y
Reshape_7/shapeConst*
valueB:@*
dtype0*
_output_shapes
:
a
	Reshape_7Reshapesplit:1Reshape_7/shape*
T0*
Tshape0*
_output_shapes
:@
`
Reshape_8/shapeConst*
valueB"@   @   *
dtype0*
_output_shapes
:
e
	Reshape_8Reshapesplit:2Reshape_8/shape*
T0*
Tshape0*
_output_shapes

:@@
Y
Reshape_9/shapeConst*
valueB:@*
dtype0*
_output_shapes
:
a
	Reshape_9Reshapesplit:3Reshape_9/shape*
_output_shapes
:@*
T0*
Tshape0
a
Reshape_10/shapeConst*
_output_shapes
:*
valueB"@      *
dtype0
g

Reshape_10Reshapesplit:4Reshape_10/shape*
_output_shapes

:@*
T0*
Tshape0
Z
Reshape_11/shapeConst*
valueB:*
dtype0*
_output_shapes
:
c

Reshape_11Reshapesplit:5Reshape_11/shape*
T0*
Tshape0*
_output_shapes
:
ђ
beta1_power/initial_valueConst*
_output_shapes
: *
valueB
 *fff?* 
_class
loc:@pi/dense/bias*
dtype0
Љ
beta1_power
VariableV2*
	container *
shape: *
dtype0*
_output_shapes
: *
shared_name * 
_class
loc:@pi/dense/bias
░
beta1_power/AssignAssignbeta1_powerbeta1_power/initial_value* 
_class
loc:@pi/dense/bias*
validate_shape(*
_output_shapes
: *
use_locking(*
T0
l
beta1_power/readIdentitybeta1_power*
T0* 
_class
loc:@pi/dense/bias*
_output_shapes
: 
ђ
beta2_power/initial_valueConst*
valueB
 *wЙ?* 
_class
loc:@pi/dense/bias*
dtype0*
_output_shapes
: 
Љ
beta2_power
VariableV2*
	container *
shape: *
dtype0*
_output_shapes
: *
shared_name * 
_class
loc:@pi/dense/bias
░
beta2_power/AssignAssignbeta2_powerbeta2_power/initial_value*
use_locking(*
T0* 
_class
loc:@pi/dense/bias*
validate_shape(*
_output_shapes
: 
l
beta2_power/readIdentitybeta2_power*
_output_shapes
: *
T0* 
_class
loc:@pi/dense/bias
Ф
6pi/dense/kernel/Adam/Initializer/zeros/shape_as_tensorConst*"
_class
loc:@pi/dense/kernel*
valueB"Є   @   *
dtype0*
_output_shapes
:
Ћ
,pi/dense/kernel/Adam/Initializer/zeros/ConstConst*
dtype0*
_output_shapes
: *"
_class
loc:@pi/dense/kernel*
valueB
 *    
З
&pi/dense/kernel/Adam/Initializer/zerosFill6pi/dense/kernel/Adam/Initializer/zeros/shape_as_tensor,pi/dense/kernel/Adam/Initializer/zeros/Const*
T0*"
_class
loc:@pi/dense/kernel*

index_type0*
_output_shapes
:	Є@
«
pi/dense/kernel/Adam
VariableV2*
shared_name *"
_class
loc:@pi/dense/kernel*
	container *
shape:	Є@*
dtype0*
_output_shapes
:	Є@
┌
pi/dense/kernel/Adam/AssignAssignpi/dense/kernel/Adam&pi/dense/kernel/Adam/Initializer/zeros*
use_locking(*
T0*"
_class
loc:@pi/dense/kernel*
validate_shape(*
_output_shapes
:	Є@
Ѕ
pi/dense/kernel/Adam/readIdentitypi/dense/kernel/Adam*
_output_shapes
:	Є@*
T0*"
_class
loc:@pi/dense/kernel
Г
8pi/dense/kernel/Adam_1/Initializer/zeros/shape_as_tensorConst*
dtype0*
_output_shapes
:*"
_class
loc:@pi/dense/kernel*
valueB"Є   @   
Ќ
.pi/dense/kernel/Adam_1/Initializer/zeros/ConstConst*"
_class
loc:@pi/dense/kernel*
valueB
 *    *
dtype0*
_output_shapes
: 
Щ
(pi/dense/kernel/Adam_1/Initializer/zerosFill8pi/dense/kernel/Adam_1/Initializer/zeros/shape_as_tensor.pi/dense/kernel/Adam_1/Initializer/zeros/Const*
_output_shapes
:	Є@*
T0*"
_class
loc:@pi/dense/kernel*

index_type0
░
pi/dense/kernel/Adam_1
VariableV2*
dtype0*
_output_shapes
:	Є@*
shared_name *"
_class
loc:@pi/dense/kernel*
	container *
shape:	Є@
Я
pi/dense/kernel/Adam_1/AssignAssignpi/dense/kernel/Adam_1(pi/dense/kernel/Adam_1/Initializer/zeros*
use_locking(*
T0*"
_class
loc:@pi/dense/kernel*
validate_shape(*
_output_shapes
:	Є@
Ї
pi/dense/kernel/Adam_1/readIdentitypi/dense/kernel/Adam_1*"
_class
loc:@pi/dense/kernel*
_output_shapes
:	Є@*
T0
Њ
$pi/dense/bias/Adam/Initializer/zerosConst* 
_class
loc:@pi/dense/bias*
valueB@*    *
dtype0*
_output_shapes
:@
а
pi/dense/bias/Adam
VariableV2*
shared_name * 
_class
loc:@pi/dense/bias*
	container *
shape:@*
dtype0*
_output_shapes
:@
═
pi/dense/bias/Adam/AssignAssignpi/dense/bias/Adam$pi/dense/bias/Adam/Initializer/zeros*
use_locking(*
T0* 
_class
loc:@pi/dense/bias*
validate_shape(*
_output_shapes
:@
~
pi/dense/bias/Adam/readIdentitypi/dense/bias/Adam* 
_class
loc:@pi/dense/bias*
_output_shapes
:@*
T0
Ћ
&pi/dense/bias/Adam_1/Initializer/zerosConst* 
_class
loc:@pi/dense/bias*
valueB@*    *
dtype0*
_output_shapes
:@
б
pi/dense/bias/Adam_1
VariableV2*
shape:@*
dtype0*
_output_shapes
:@*
shared_name * 
_class
loc:@pi/dense/bias*
	container 
М
pi/dense/bias/Adam_1/AssignAssignpi/dense/bias/Adam_1&pi/dense/bias/Adam_1/Initializer/zeros*
use_locking(*
T0* 
_class
loc:@pi/dense/bias*
validate_shape(*
_output_shapes
:@
ѓ
pi/dense/bias/Adam_1/readIdentitypi/dense/bias/Adam_1*
_output_shapes
:@*
T0* 
_class
loc:@pi/dense/bias
»
8pi/dense_1/kernel/Adam/Initializer/zeros/shape_as_tensorConst*$
_class
loc:@pi/dense_1/kernel*
valueB"@   @   *
dtype0*
_output_shapes
:
Ў
.pi/dense_1/kernel/Adam/Initializer/zeros/ConstConst*$
_class
loc:@pi/dense_1/kernel*
valueB
 *    *
dtype0*
_output_shapes
: 
ч
(pi/dense_1/kernel/Adam/Initializer/zerosFill8pi/dense_1/kernel/Adam/Initializer/zeros/shape_as_tensor.pi/dense_1/kernel/Adam/Initializer/zeros/Const*
T0*$
_class
loc:@pi/dense_1/kernel*

index_type0*
_output_shapes

:@@
░
pi/dense_1/kernel/Adam
VariableV2*$
_class
loc:@pi/dense_1/kernel*
	container *
shape
:@@*
dtype0*
_output_shapes

:@@*
shared_name 
р
pi/dense_1/kernel/Adam/AssignAssignpi/dense_1/kernel/Adam(pi/dense_1/kernel/Adam/Initializer/zeros*
use_locking(*
T0*$
_class
loc:@pi/dense_1/kernel*
validate_shape(*
_output_shapes

:@@
ј
pi/dense_1/kernel/Adam/readIdentitypi/dense_1/kernel/Adam*
T0*$
_class
loc:@pi/dense_1/kernel*
_output_shapes

:@@
▒
:pi/dense_1/kernel/Adam_1/Initializer/zeros/shape_as_tensorConst*$
_class
loc:@pi/dense_1/kernel*
valueB"@   @   *
dtype0*
_output_shapes
:
Џ
0pi/dense_1/kernel/Adam_1/Initializer/zeros/ConstConst*
_output_shapes
: *$
_class
loc:@pi/dense_1/kernel*
valueB
 *    *
dtype0
Ђ
*pi/dense_1/kernel/Adam_1/Initializer/zerosFill:pi/dense_1/kernel/Adam_1/Initializer/zeros/shape_as_tensor0pi/dense_1/kernel/Adam_1/Initializer/zeros/Const*
T0*$
_class
loc:@pi/dense_1/kernel*

index_type0*
_output_shapes

:@@
▓
pi/dense_1/kernel/Adam_1
VariableV2*
shared_name *$
_class
loc:@pi/dense_1/kernel*
	container *
shape
:@@*
dtype0*
_output_shapes

:@@
у
pi/dense_1/kernel/Adam_1/AssignAssignpi/dense_1/kernel/Adam_1*pi/dense_1/kernel/Adam_1/Initializer/zeros*
use_locking(*
T0*$
_class
loc:@pi/dense_1/kernel*
validate_shape(*
_output_shapes

:@@
њ
pi/dense_1/kernel/Adam_1/readIdentitypi/dense_1/kernel/Adam_1*
T0*$
_class
loc:@pi/dense_1/kernel*
_output_shapes

:@@
Ќ
&pi/dense_1/bias/Adam/Initializer/zerosConst*"
_class
loc:@pi/dense_1/bias*
valueB@*    *
dtype0*
_output_shapes
:@
ц
pi/dense_1/bias/Adam
VariableV2*
shape:@*
dtype0*
_output_shapes
:@*
shared_name *"
_class
loc:@pi/dense_1/bias*
	container 
Н
pi/dense_1/bias/Adam/AssignAssignpi/dense_1/bias/Adam&pi/dense_1/bias/Adam/Initializer/zeros*
validate_shape(*
_output_shapes
:@*
use_locking(*
T0*"
_class
loc:@pi/dense_1/bias
ё
pi/dense_1/bias/Adam/readIdentitypi/dense_1/bias/Adam*
T0*"
_class
loc:@pi/dense_1/bias*
_output_shapes
:@
Ў
(pi/dense_1/bias/Adam_1/Initializer/zerosConst*"
_class
loc:@pi/dense_1/bias*
valueB@*    *
dtype0*
_output_shapes
:@
д
pi/dense_1/bias/Adam_1
VariableV2*
shape:@*
dtype0*
_output_shapes
:@*
shared_name *"
_class
loc:@pi/dense_1/bias*
	container 
█
pi/dense_1/bias/Adam_1/AssignAssignpi/dense_1/bias/Adam_1(pi/dense_1/bias/Adam_1/Initializer/zeros*
use_locking(*
T0*"
_class
loc:@pi/dense_1/bias*
validate_shape(*
_output_shapes
:@
ѕ
pi/dense_1/bias/Adam_1/readIdentitypi/dense_1/bias/Adam_1*
T0*"
_class
loc:@pi/dense_1/bias*
_output_shapes
:@
Б
(pi/dense_2/kernel/Adam/Initializer/zerosConst*$
_class
loc:@pi/dense_2/kernel*
valueB@*    *
dtype0*
_output_shapes

:@
░
pi/dense_2/kernel/Adam
VariableV2*
dtype0*
_output_shapes

:@*
shared_name *$
_class
loc:@pi/dense_2/kernel*
	container *
shape
:@
р
pi/dense_2/kernel/Adam/AssignAssignpi/dense_2/kernel/Adam(pi/dense_2/kernel/Adam/Initializer/zeros*
use_locking(*
T0*$
_class
loc:@pi/dense_2/kernel*
validate_shape(*
_output_shapes

:@
ј
pi/dense_2/kernel/Adam/readIdentitypi/dense_2/kernel/Adam*
T0*$
_class
loc:@pi/dense_2/kernel*
_output_shapes

:@
Ц
*pi/dense_2/kernel/Adam_1/Initializer/zerosConst*
_output_shapes

:@*$
_class
loc:@pi/dense_2/kernel*
valueB@*    *
dtype0
▓
pi/dense_2/kernel/Adam_1
VariableV2*
dtype0*
_output_shapes

:@*
shared_name *$
_class
loc:@pi/dense_2/kernel*
	container *
shape
:@
у
pi/dense_2/kernel/Adam_1/AssignAssignpi/dense_2/kernel/Adam_1*pi/dense_2/kernel/Adam_1/Initializer/zeros*
use_locking(*
T0*$
_class
loc:@pi/dense_2/kernel*
validate_shape(*
_output_shapes

:@
њ
pi/dense_2/kernel/Adam_1/readIdentitypi/dense_2/kernel/Adam_1*
T0*$
_class
loc:@pi/dense_2/kernel*
_output_shapes

:@
Ќ
&pi/dense_2/bias/Adam/Initializer/zerosConst*"
_class
loc:@pi/dense_2/bias*
valueB*    *
dtype0*
_output_shapes
:
ц
pi/dense_2/bias/Adam
VariableV2*
shape:*
dtype0*
_output_shapes
:*
shared_name *"
_class
loc:@pi/dense_2/bias*
	container 
Н
pi/dense_2/bias/Adam/AssignAssignpi/dense_2/bias/Adam&pi/dense_2/bias/Adam/Initializer/zeros*
use_locking(*
T0*"
_class
loc:@pi/dense_2/bias*
validate_shape(*
_output_shapes
:
ё
pi/dense_2/bias/Adam/readIdentitypi/dense_2/bias/Adam*
T0*"
_class
loc:@pi/dense_2/bias*
_output_shapes
:
Ў
(pi/dense_2/bias/Adam_1/Initializer/zerosConst*
_output_shapes
:*"
_class
loc:@pi/dense_2/bias*
valueB*    *
dtype0
д
pi/dense_2/bias/Adam_1
VariableV2*
dtype0*
_output_shapes
:*
shared_name *"
_class
loc:@pi/dense_2/bias*
	container *
shape:
█
pi/dense_2/bias/Adam_1/AssignAssignpi/dense_2/bias/Adam_1(pi/dense_2/bias/Adam_1/Initializer/zeros*
validate_shape(*
_output_shapes
:*
use_locking(*
T0*"
_class
loc:@pi/dense_2/bias
ѕ
pi/dense_2/bias/Adam_1/readIdentitypi/dense_2/bias/Adam_1*
T0*"
_class
loc:@pi/dense_2/bias*
_output_shapes
:
W
Adam/learning_rateConst*
valueB
 *RIЮ9*
dtype0*
_output_shapes
: 
O

Adam/beta1Const*
_output_shapes
: *
valueB
 *fff?*
dtype0
O

Adam/beta2Const*
dtype0*
_output_shapes
: *
valueB
 *wЙ?
Q
Adam/epsilonConst*
valueB
 *w╠+2*
dtype0*
_output_shapes
: 
¤
%Adam/update_pi/dense/kernel/ApplyAdam	ApplyAdampi/dense/kernelpi/dense/kernel/Adampi/dense/kernel/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilon	Reshape_6*
_output_shapes
:	Є@*
use_locking( *
T0*"
_class
loc:@pi/dense/kernel*
use_nesterov( 
└
#Adam/update_pi/dense/bias/ApplyAdam	ApplyAdampi/dense/biaspi/dense/bias/Adampi/dense/bias/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilon	Reshape_7*
use_locking( *
T0* 
_class
loc:@pi/dense/bias*
use_nesterov( *
_output_shapes
:@
п
'Adam/update_pi/dense_1/kernel/ApplyAdam	ApplyAdampi/dense_1/kernelpi/dense_1/kernel/Adampi/dense_1/kernel/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilon	Reshape_8*
use_locking( *
T0*$
_class
loc:@pi/dense_1/kernel*
use_nesterov( *
_output_shapes

:@@
╩
%Adam/update_pi/dense_1/bias/ApplyAdam	ApplyAdampi/dense_1/biaspi/dense_1/bias/Adampi/dense_1/bias/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilon	Reshape_9*
use_nesterov( *
_output_shapes
:@*
use_locking( *
T0*"
_class
loc:@pi/dense_1/bias
┘
'Adam/update_pi/dense_2/kernel/ApplyAdam	ApplyAdampi/dense_2/kernelpi/dense_2/kernel/Adampi/dense_2/kernel/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilon
Reshape_10*
use_nesterov( *
_output_shapes

:@*
use_locking( *
T0*$
_class
loc:@pi/dense_2/kernel
╦
%Adam/update_pi/dense_2/bias/ApplyAdam	ApplyAdampi/dense_2/biaspi/dense_2/bias/Adampi/dense_2/bias/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilon
Reshape_11*"
_class
loc:@pi/dense_2/bias*
use_nesterov( *
_output_shapes
:*
use_locking( *
T0
Р
Adam/mulMulbeta1_power/read
Adam/beta1$^Adam/update_pi/dense/bias/ApplyAdam&^Adam/update_pi/dense/kernel/ApplyAdam&^Adam/update_pi/dense_1/bias/ApplyAdam(^Adam/update_pi/dense_1/kernel/ApplyAdam&^Adam/update_pi/dense_2/bias/ApplyAdam(^Adam/update_pi/dense_2/kernel/ApplyAdam*
T0* 
_class
loc:@pi/dense/bias*
_output_shapes
: 
ў
Adam/AssignAssignbeta1_powerAdam/mul*
use_locking( *
T0* 
_class
loc:@pi/dense/bias*
validate_shape(*
_output_shapes
: 
С

Adam/mul_1Mulbeta2_power/read
Adam/beta2$^Adam/update_pi/dense/bias/ApplyAdam&^Adam/update_pi/dense/kernel/ApplyAdam&^Adam/update_pi/dense_1/bias/ApplyAdam(^Adam/update_pi/dense_1/kernel/ApplyAdam&^Adam/update_pi/dense_2/bias/ApplyAdam(^Adam/update_pi/dense_2/kernel/ApplyAdam*
_output_shapes
: *
T0* 
_class
loc:@pi/dense/bias
ю
Adam/Assign_1Assignbeta2_power
Adam/mul_1*
use_locking( *
T0* 
_class
loc:@pi/dense/bias*
validate_shape(*
_output_shapes
: 
ю
AdamNoOp^Adam/Assign^Adam/Assign_1$^Adam/update_pi/dense/bias/ApplyAdam&^Adam/update_pi/dense/kernel/ApplyAdam&^Adam/update_pi/dense_1/bias/ApplyAdam(^Adam/update_pi/dense_1/kernel/ApplyAdam&^Adam/update_pi/dense_2/bias/ApplyAdam(^Adam/update_pi/dense_2/kernel/ApplyAdam
j
Reshape_12/shapeConst^Adam*
_output_shapes
:*
valueB:
         *
dtype0
q

Reshape_12Reshapepi/dense/kernel/readReshape_12/shape*
T0*
Tshape0*
_output_shapes	
:└C
j
Reshape_13/shapeConst^Adam*
valueB:
         *
dtype0*
_output_shapes
:
n

Reshape_13Reshapepi/dense/bias/readReshape_13/shape*
_output_shapes
:@*
T0*
Tshape0
j
Reshape_14/shapeConst^Adam*
valueB:
         *
dtype0*
_output_shapes
:
s

Reshape_14Reshapepi/dense_1/kernel/readReshape_14/shape*
T0*
Tshape0*
_output_shapes	
:ђ 
j
Reshape_15/shapeConst^Adam*
valueB:
         *
dtype0*
_output_shapes
:
p

Reshape_15Reshapepi/dense_1/bias/readReshape_15/shape*
Tshape0*
_output_shapes
:@*
T0
j
Reshape_16/shapeConst^Adam*
_output_shapes
:*
valueB:
         *
dtype0
s

Reshape_16Reshapepi/dense_2/kernel/readReshape_16/shape*
T0*
Tshape0*
_output_shapes	
:ђ
j
Reshape_17/shapeConst^Adam*
valueB:
         *
dtype0*
_output_shapes
:
p

Reshape_17Reshapepi/dense_2/bias/readReshape_17/shape*
_output_shapes
:*
T0*
Tshape0
V
concat_1/axisConst^Adam*
value	B : *
dtype0*
_output_shapes
: 
д
concat_1ConcatV2
Reshape_12
Reshape_13
Reshape_14
Reshape_15
Reshape_16
Reshape_17concat_1/axis*
_output_shapes	
:─f*

Tidx0*
T0*
N
h
PyFunc_1PyFuncconcat_1*
token
pyfunc_1*
_output_shapes
:*
Tin
2*
Tout
2
o
Const_6Const^Adam*-
value$B""└!  @      @         *
dtype0*
_output_shapes
:
Z
split_1/split_dimConst^Adam*
value	B : *
dtype0*
_output_shapes
: 
І
split_1SplitVPyFunc_1Const_6split_1/split_dim*
T0*,
_output_shapes
::::::*
	num_split*

Tlen0
h
Reshape_18/shapeConst^Adam*
valueB"Є   @   *
dtype0*
_output_shapes
:
h

Reshape_18Reshapesplit_1Reshape_18/shape*
T0*
Tshape0*
_output_shapes
:	Є@
a
Reshape_19/shapeConst^Adam*
valueB:@*
dtype0*
_output_shapes
:
e

Reshape_19Reshape	split_1:1Reshape_19/shape*
T0*
Tshape0*
_output_shapes
:@
h
Reshape_20/shapeConst^Adam*
valueB"@   @   *
dtype0*
_output_shapes
:
i

Reshape_20Reshape	split_1:2Reshape_20/shape*
T0*
Tshape0*
_output_shapes

:@@
a
Reshape_21/shapeConst^Adam*
valueB:@*
dtype0*
_output_shapes
:
e

Reshape_21Reshape	split_1:3Reshape_21/shape*
_output_shapes
:@*
T0*
Tshape0
h
Reshape_22/shapeConst^Adam*
_output_shapes
:*
valueB"@      *
dtype0
i

Reshape_22Reshape	split_1:4Reshape_22/shape*
T0*
Tshape0*
_output_shapes

:@
a
Reshape_23/shapeConst^Adam*
valueB:*
dtype0*
_output_shapes
:
e

Reshape_23Reshape	split_1:5Reshape_23/shape*
T0*
Tshape0*
_output_shapes
:
ц
AssignAssignpi/dense/kernel
Reshape_18*
_output_shapes
:	Є@*
use_locking(*
T0*"
_class
loc:@pi/dense/kernel*
validate_shape(
Ю
Assign_1Assignpi/dense/bias
Reshape_19*
use_locking(*
T0* 
_class
loc:@pi/dense/bias*
validate_shape(*
_output_shapes
:@
Е
Assign_2Assignpi/dense_1/kernel
Reshape_20*
_output_shapes

:@@*
use_locking(*
T0*$
_class
loc:@pi/dense_1/kernel*
validate_shape(
А
Assign_3Assignpi/dense_1/bias
Reshape_21*
use_locking(*
T0*"
_class
loc:@pi/dense_1/bias*
validate_shape(*
_output_shapes
:@
Е
Assign_4Assignpi/dense_2/kernel
Reshape_22*
_output_shapes

:@*
use_locking(*
T0*$
_class
loc:@pi/dense_2/kernel*
validate_shape(
А
Assign_5Assignpi/dense_2/bias
Reshape_23*
T0*"
_class
loc:@pi/dense_2/bias*
validate_shape(*
_output_shapes
:*
use_locking(
Y

group_depsNoOp^Adam^Assign	^Assign_1	^Assign_2	^Assign_3	^Assign_4	^Assign_5
(
group_deps_1NoOp^Adam^group_deps
T
gradients_1/ShapeConst*
dtype0*
_output_shapes
: *
valueB 
Z
gradients_1/grad_ys_0Const*
valueB
 *  ђ?*
dtype0*
_output_shapes
: 
u
gradients_1/FillFillgradients_1/Shapegradients_1/grad_ys_0*
_output_shapes
: *
T0*

index_type0
o
%gradients_1/Mean_1_grad/Reshape/shapeConst*
valueB:*
dtype0*
_output_shapes
:
ќ
gradients_1/Mean_1_grad/ReshapeReshapegradients_1/Fill%gradients_1/Mean_1_grad/Reshape/shape*
_output_shapes
:*
T0*
Tshape0
`
gradients_1/Mean_1_grad/ShapeShapepow*
_output_shapes
:*
T0*
out_type0
ц
gradients_1/Mean_1_grad/TileTilegradients_1/Mean_1_grad/Reshapegradients_1/Mean_1_grad/Shape*

Tmultiples0*
T0*#
_output_shapes
:         
b
gradients_1/Mean_1_grad/Shape_1Shapepow*
_output_shapes
:*
T0*
out_type0
b
gradients_1/Mean_1_grad/Shape_2Const*
valueB *
dtype0*
_output_shapes
: 
g
gradients_1/Mean_1_grad/ConstConst*
valueB: *
dtype0*
_output_shapes
:
б
gradients_1/Mean_1_grad/ProdProdgradients_1/Mean_1_grad/Shape_1gradients_1/Mean_1_grad/Const*
T0*
_output_shapes
: *
	keep_dims( *

Tidx0
i
gradients_1/Mean_1_grad/Const_1Const*
valueB: *
dtype0*
_output_shapes
:
д
gradients_1/Mean_1_grad/Prod_1Prodgradients_1/Mean_1_grad/Shape_2gradients_1/Mean_1_grad/Const_1*
T0*
_output_shapes
: *
	keep_dims( *

Tidx0
c
!gradients_1/Mean_1_grad/Maximum/yConst*
_output_shapes
: *
value	B :*
dtype0
ј
gradients_1/Mean_1_grad/MaximumMaximumgradients_1/Mean_1_grad/Prod_1!gradients_1/Mean_1_grad/Maximum/y*
T0*
_output_shapes
: 
ї
 gradients_1/Mean_1_grad/floordivFloorDivgradients_1/Mean_1_grad/Prodgradients_1/Mean_1_grad/Maximum*
_output_shapes
: *
T0
є
gradients_1/Mean_1_grad/CastCast gradients_1/Mean_1_grad/floordiv*
Truncate( *
_output_shapes
: *

DstT0*

SrcT0
ћ
gradients_1/Mean_1_grad/truedivRealDivgradients_1/Mean_1_grad/Tilegradients_1/Mean_1_grad/Cast*
T0*#
_output_shapes
:         
_
gradients_1/pow_grad/ShapeShapesub_1*
_output_shapes
:*
T0*
out_type0
_
gradients_1/pow_grad/Shape_1Shapepow/y*
T0*
out_type0*
_output_shapes
: 
║
*gradients_1/pow_grad/BroadcastGradientArgsBroadcastGradientArgsgradients_1/pow_grad/Shapegradients_1/pow_grad/Shape_1*
T0*2
_output_shapes 
:         :         
u
gradients_1/pow_grad/mulMulgradients_1/Mean_1_grad/truedivpow/y*#
_output_shapes
:         *
T0
_
gradients_1/pow_grad/sub/yConst*
valueB
 *  ђ?*
dtype0*
_output_shapes
: 
c
gradients_1/pow_grad/subSubpow/ygradients_1/pow_grad/sub/y*
_output_shapes
: *
T0
n
gradients_1/pow_grad/PowPowsub_1gradients_1/pow_grad/sub*
T0*#
_output_shapes
:         
Ѓ
gradients_1/pow_grad/mul_1Mulgradients_1/pow_grad/mulgradients_1/pow_grad/Pow*
T0*#
_output_shapes
:         
Д
gradients_1/pow_grad/SumSumgradients_1/pow_grad/mul_1*gradients_1/pow_grad/BroadcastGradientArgs*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
Ў
gradients_1/pow_grad/ReshapeReshapegradients_1/pow_grad/Sumgradients_1/pow_grad/Shape*
T0*
Tshape0*#
_output_shapes
:         
c
gradients_1/pow_grad/Greater/yConst*
valueB
 *    *
dtype0*
_output_shapes
: 
|
gradients_1/pow_grad/GreaterGreatersub_1gradients_1/pow_grad/Greater/y*
T0*#
_output_shapes
:         
i
$gradients_1/pow_grad/ones_like/ShapeShapesub_1*
T0*
out_type0*
_output_shapes
:
i
$gradients_1/pow_grad/ones_like/ConstConst*
valueB
 *  ђ?*
dtype0*
_output_shapes
: 
▓
gradients_1/pow_grad/ones_likeFill$gradients_1/pow_grad/ones_like/Shape$gradients_1/pow_grad/ones_like/Const*
T0*

index_type0*#
_output_shapes
:         
ў
gradients_1/pow_grad/SelectSelectgradients_1/pow_grad/Greatersub_1gradients_1/pow_grad/ones_like*
T0*#
_output_shapes
:         
j
gradients_1/pow_grad/LogLoggradients_1/pow_grad/Select*
T0*#
_output_shapes
:         
a
gradients_1/pow_grad/zeros_like	ZerosLikesub_1*
T0*#
_output_shapes
:         
«
gradients_1/pow_grad/Select_1Selectgradients_1/pow_grad/Greatergradients_1/pow_grad/Loggradients_1/pow_grad/zeros_like*#
_output_shapes
:         *
T0
u
gradients_1/pow_grad/mul_2Mulgradients_1/Mean_1_grad/truedivpow*#
_output_shapes
:         *
T0
і
gradients_1/pow_grad/mul_3Mulgradients_1/pow_grad/mul_2gradients_1/pow_grad/Select_1*
T0*#
_output_shapes
:         
Ф
gradients_1/pow_grad/Sum_1Sumgradients_1/pow_grad/mul_3,gradients_1/pow_grad/BroadcastGradientArgs:1*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
њ
gradients_1/pow_grad/Reshape_1Reshapegradients_1/pow_grad/Sum_1gradients_1/pow_grad/Shape_1*
Tshape0*
_output_shapes
: *
T0
m
%gradients_1/pow_grad/tuple/group_depsNoOp^gradients_1/pow_grad/Reshape^gradients_1/pow_grad/Reshape_1
я
-gradients_1/pow_grad/tuple/control_dependencyIdentitygradients_1/pow_grad/Reshape&^gradients_1/pow_grad/tuple/group_deps*/
_class%
#!loc:@gradients_1/pow_grad/Reshape*#
_output_shapes
:         *
T0
О
/gradients_1/pow_grad/tuple/control_dependency_1Identitygradients_1/pow_grad/Reshape_1&^gradients_1/pow_grad/tuple/group_deps*
T0*1
_class'
%#loc:@gradients_1/pow_grad/Reshape_1*
_output_shapes
: 
i
gradients_1/sub_1_grad/ShapeShapePlaceholder_3*
T0*
out_type0*
_output_shapes
:
g
gradients_1/sub_1_grad/Shape_1Shape	v/Squeeze*
out_type0*
_output_shapes
:*
T0
└
,gradients_1/sub_1_grad/BroadcastGradientArgsBroadcastGradientArgsgradients_1/sub_1_grad/Shapegradients_1/sub_1_grad/Shape_1*2
_output_shapes 
:         :         *
T0
Й
gradients_1/sub_1_grad/SumSum-gradients_1/pow_grad/tuple/control_dependency,gradients_1/sub_1_grad/BroadcastGradientArgs*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0
Ъ
gradients_1/sub_1_grad/ReshapeReshapegradients_1/sub_1_grad/Sumgradients_1/sub_1_grad/Shape*#
_output_shapes
:         *
T0*
Tshape0
~
gradients_1/sub_1_grad/NegNeg-gradients_1/pow_grad/tuple/control_dependency*#
_output_shapes
:         *
T0
»
gradients_1/sub_1_grad/Sum_1Sumgradients_1/sub_1_grad/Neg.gradients_1/sub_1_grad/BroadcastGradientArgs:1*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
Ц
 gradients_1/sub_1_grad/Reshape_1Reshapegradients_1/sub_1_grad/Sum_1gradients_1/sub_1_grad/Shape_1*
T0*
Tshape0*#
_output_shapes
:         
s
'gradients_1/sub_1_grad/tuple/group_depsNoOp^gradients_1/sub_1_grad/Reshape!^gradients_1/sub_1_grad/Reshape_1
Т
/gradients_1/sub_1_grad/tuple/control_dependencyIdentitygradients_1/sub_1_grad/Reshape(^gradients_1/sub_1_grad/tuple/group_deps*
T0*1
_class'
%#loc:@gradients_1/sub_1_grad/Reshape*#
_output_shapes
:         
В
1gradients_1/sub_1_grad/tuple/control_dependency_1Identity gradients_1/sub_1_grad/Reshape_1(^gradients_1/sub_1_grad/tuple/group_deps*
T0*3
_class)
'%loc:@gradients_1/sub_1_grad/Reshape_1*#
_output_shapes
:         
q
 gradients_1/v/Squeeze_grad/ShapeShapev/dense_2/BiasAdd*
out_type0*
_output_shapes
:*
T0
┬
"gradients_1/v/Squeeze_grad/ReshapeReshape1gradients_1/sub_1_grad/tuple/control_dependency_1 gradients_1/v/Squeeze_grad/Shape*
T0*
Tshape0*'
_output_shapes
:         
Ю
.gradients_1/v/dense_2/BiasAdd_grad/BiasAddGradBiasAddGrad"gradients_1/v/Squeeze_grad/Reshape*
_output_shapes
:*
T0*
data_formatNHWC
Љ
3gradients_1/v/dense_2/BiasAdd_grad/tuple/group_depsNoOp#^gradients_1/v/Squeeze_grad/Reshape/^gradients_1/v/dense_2/BiasAdd_grad/BiasAddGrad
і
;gradients_1/v/dense_2/BiasAdd_grad/tuple/control_dependencyIdentity"gradients_1/v/Squeeze_grad/Reshape4^gradients_1/v/dense_2/BiasAdd_grad/tuple/group_deps*
T0*5
_class+
)'loc:@gradients_1/v/Squeeze_grad/Reshape*'
_output_shapes
:         
Ќ
=gradients_1/v/dense_2/BiasAdd_grad/tuple/control_dependency_1Identity.gradients_1/v/dense_2/BiasAdd_grad/BiasAddGrad4^gradients_1/v/dense_2/BiasAdd_grad/tuple/group_deps*
T0*A
_class7
53loc:@gradients_1/v/dense_2/BiasAdd_grad/BiasAddGrad*
_output_shapes
:
я
(gradients_1/v/dense_2/MatMul_grad/MatMulMatMul;gradients_1/v/dense_2/BiasAdd_grad/tuple/control_dependencyv/dense_2/kernel/read*'
_output_shapes
:         @*
transpose_a( *
transpose_b(*
T0
л
*gradients_1/v/dense_2/MatMul_grad/MatMul_1MatMulv/dense_1/Tanh;gradients_1/v/dense_2/BiasAdd_grad/tuple/control_dependency*
T0*
_output_shapes

:@*
transpose_a(*
transpose_b( 
њ
2gradients_1/v/dense_2/MatMul_grad/tuple/group_depsNoOp)^gradients_1/v/dense_2/MatMul_grad/MatMul+^gradients_1/v/dense_2/MatMul_grad/MatMul_1
ћ
:gradients_1/v/dense_2/MatMul_grad/tuple/control_dependencyIdentity(gradients_1/v/dense_2/MatMul_grad/MatMul3^gradients_1/v/dense_2/MatMul_grad/tuple/group_deps*
T0*;
_class1
/-loc:@gradients_1/v/dense_2/MatMul_grad/MatMul*'
_output_shapes
:         @
Љ
<gradients_1/v/dense_2/MatMul_grad/tuple/control_dependency_1Identity*gradients_1/v/dense_2/MatMul_grad/MatMul_13^gradients_1/v/dense_2/MatMul_grad/tuple/group_deps*
_output_shapes

:@*
T0*=
_class3
1/loc:@gradients_1/v/dense_2/MatMul_grad/MatMul_1
▓
(gradients_1/v/dense_1/Tanh_grad/TanhGradTanhGradv/dense_1/Tanh:gradients_1/v/dense_2/MatMul_grad/tuple/control_dependency*
T0*'
_output_shapes
:         @
Б
.gradients_1/v/dense_1/BiasAdd_grad/BiasAddGradBiasAddGrad(gradients_1/v/dense_1/Tanh_grad/TanhGrad*
T0*
data_formatNHWC*
_output_shapes
:@
Ќ
3gradients_1/v/dense_1/BiasAdd_grad/tuple/group_depsNoOp/^gradients_1/v/dense_1/BiasAdd_grad/BiasAddGrad)^gradients_1/v/dense_1/Tanh_grad/TanhGrad
ќ
;gradients_1/v/dense_1/BiasAdd_grad/tuple/control_dependencyIdentity(gradients_1/v/dense_1/Tanh_grad/TanhGrad4^gradients_1/v/dense_1/BiasAdd_grad/tuple/group_deps*'
_output_shapes
:         @*
T0*;
_class1
/-loc:@gradients_1/v/dense_1/Tanh_grad/TanhGrad
Ќ
=gradients_1/v/dense_1/BiasAdd_grad/tuple/control_dependency_1Identity.gradients_1/v/dense_1/BiasAdd_grad/BiasAddGrad4^gradients_1/v/dense_1/BiasAdd_grad/tuple/group_deps*
T0*A
_class7
53loc:@gradients_1/v/dense_1/BiasAdd_grad/BiasAddGrad*
_output_shapes
:@
я
(gradients_1/v/dense_1/MatMul_grad/MatMulMatMul;gradients_1/v/dense_1/BiasAdd_grad/tuple/control_dependencyv/dense_1/kernel/read*
T0*'
_output_shapes
:         @*
transpose_a( *
transpose_b(
╬
*gradients_1/v/dense_1/MatMul_grad/MatMul_1MatMulv/dense/Tanh;gradients_1/v/dense_1/BiasAdd_grad/tuple/control_dependency*
_output_shapes

:@@*
transpose_a(*
transpose_b( *
T0
њ
2gradients_1/v/dense_1/MatMul_grad/tuple/group_depsNoOp)^gradients_1/v/dense_1/MatMul_grad/MatMul+^gradients_1/v/dense_1/MatMul_grad/MatMul_1
ћ
:gradients_1/v/dense_1/MatMul_grad/tuple/control_dependencyIdentity(gradients_1/v/dense_1/MatMul_grad/MatMul3^gradients_1/v/dense_1/MatMul_grad/tuple/group_deps*
T0*;
_class1
/-loc:@gradients_1/v/dense_1/MatMul_grad/MatMul*'
_output_shapes
:         @
Љ
<gradients_1/v/dense_1/MatMul_grad/tuple/control_dependency_1Identity*gradients_1/v/dense_1/MatMul_grad/MatMul_13^gradients_1/v/dense_1/MatMul_grad/tuple/group_deps*
T0*=
_class3
1/loc:@gradients_1/v/dense_1/MatMul_grad/MatMul_1*
_output_shapes

:@@
«
&gradients_1/v/dense/Tanh_grad/TanhGradTanhGradv/dense/Tanh:gradients_1/v/dense_1/MatMul_grad/tuple/control_dependency*
T0*'
_output_shapes
:         @
Ъ
,gradients_1/v/dense/BiasAdd_grad/BiasAddGradBiasAddGrad&gradients_1/v/dense/Tanh_grad/TanhGrad*
data_formatNHWC*
_output_shapes
:@*
T0
Љ
1gradients_1/v/dense/BiasAdd_grad/tuple/group_depsNoOp-^gradients_1/v/dense/BiasAdd_grad/BiasAddGrad'^gradients_1/v/dense/Tanh_grad/TanhGrad
ј
9gradients_1/v/dense/BiasAdd_grad/tuple/control_dependencyIdentity&gradients_1/v/dense/Tanh_grad/TanhGrad2^gradients_1/v/dense/BiasAdd_grad/tuple/group_deps*9
_class/
-+loc:@gradients_1/v/dense/Tanh_grad/TanhGrad*'
_output_shapes
:         @*
T0
Ј
;gradients_1/v/dense/BiasAdd_grad/tuple/control_dependency_1Identity,gradients_1/v/dense/BiasAdd_grad/BiasAddGrad2^gradients_1/v/dense/BiasAdd_grad/tuple/group_deps*
T0*?
_class5
31loc:@gradients_1/v/dense/BiasAdd_grad/BiasAddGrad*
_output_shapes
:@
┘
&gradients_1/v/dense/MatMul_grad/MatMulMatMul9gradients_1/v/dense/BiasAdd_grad/tuple/control_dependencyv/dense/kernel/read*
T0*(
_output_shapes
:         Є*
transpose_a( *
transpose_b(
╩
(gradients_1/v/dense/MatMul_grad/MatMul_1MatMulPlaceholder9gradients_1/v/dense/BiasAdd_grad/tuple/control_dependency*
_output_shapes
:	Є@*
transpose_a(*
transpose_b( *
T0
ї
0gradients_1/v/dense/MatMul_grad/tuple/group_depsNoOp'^gradients_1/v/dense/MatMul_grad/MatMul)^gradients_1/v/dense/MatMul_grad/MatMul_1
Ї
8gradients_1/v/dense/MatMul_grad/tuple/control_dependencyIdentity&gradients_1/v/dense/MatMul_grad/MatMul1^gradients_1/v/dense/MatMul_grad/tuple/group_deps*
T0*9
_class/
-+loc:@gradients_1/v/dense/MatMul_grad/MatMul*(
_output_shapes
:         Є
і
:gradients_1/v/dense/MatMul_grad/tuple/control_dependency_1Identity(gradients_1/v/dense/MatMul_grad/MatMul_11^gradients_1/v/dense/MatMul_grad/tuple/group_deps*
T0*;
_class1
/-loc:@gradients_1/v/dense/MatMul_grad/MatMul_1*
_output_shapes
:	Є@
c
Reshape_24/shapeConst*
valueB:
         *
dtype0*
_output_shapes
:
Ќ

Reshape_24Reshape:gradients_1/v/dense/MatMul_grad/tuple/control_dependency_1Reshape_24/shape*
T0*
Tshape0*
_output_shapes	
:└C
c
Reshape_25/shapeConst*
valueB:
         *
dtype0*
_output_shapes
:
Ќ

Reshape_25Reshape;gradients_1/v/dense/BiasAdd_grad/tuple/control_dependency_1Reshape_25/shape*
_output_shapes
:@*
T0*
Tshape0
c
Reshape_26/shapeConst*
valueB:
         *
dtype0*
_output_shapes
:
Ў

Reshape_26Reshape<gradients_1/v/dense_1/MatMul_grad/tuple/control_dependency_1Reshape_26/shape*
T0*
Tshape0*
_output_shapes	
:ђ 
c
Reshape_27/shapeConst*
dtype0*
_output_shapes
:*
valueB:
         
Ў

Reshape_27Reshape=gradients_1/v/dense_1/BiasAdd_grad/tuple/control_dependency_1Reshape_27/shape*
_output_shapes
:@*
T0*
Tshape0
c
Reshape_28/shapeConst*
valueB:
         *
dtype0*
_output_shapes
:
ў

Reshape_28Reshape<gradients_1/v/dense_2/MatMul_grad/tuple/control_dependency_1Reshape_28/shape*
T0*
Tshape0*
_output_shapes
:@
c
Reshape_29/shapeConst*
dtype0*
_output_shapes
:*
valueB:
         
Ў

Reshape_29Reshape=gradients_1/v/dense_2/BiasAdd_grad/tuple/control_dependency_1Reshape_29/shape*
T0*
Tshape0*
_output_shapes
:
O
concat_2/axisConst*
value	B : *
dtype0*
_output_shapes
: 
д
concat_2ConcatV2
Reshape_24
Reshape_25
Reshape_26
Reshape_27
Reshape_28
Reshape_29concat_2/axis*
_output_shapes	
:Ђe*

Tidx0*
T0*
N
k
PyFunc_2PyFuncconcat_2*
_output_shapes	
:Ђe*
Tin
2*
Tout
2*
token
pyfunc_2
h
Const_7Const*-
value$B""└!  @      @   @      *
dtype0*
_output_shapes
:
S
split_2/split_dimConst*
value	B : *
dtype0*
_output_shapes
: 
Ў
split_2SplitVPyFunc_2Const_7split_2/split_dim*:
_output_shapes(
&:└C:@:ђ :@:@:*
	num_split*

Tlen0*
T0
a
Reshape_30/shapeConst*
valueB"Є   @   *
dtype0*
_output_shapes
:
h

Reshape_30Reshapesplit_2Reshape_30/shape*
T0*
Tshape0*
_output_shapes
:	Є@
Z
Reshape_31/shapeConst*
valueB:@*
dtype0*
_output_shapes
:
e

Reshape_31Reshape	split_2:1Reshape_31/shape*
T0*
Tshape0*
_output_shapes
:@
a
Reshape_32/shapeConst*
valueB"@   @   *
dtype0*
_output_shapes
:
i

Reshape_32Reshape	split_2:2Reshape_32/shape*
T0*
Tshape0*
_output_shapes

:@@
Z
Reshape_33/shapeConst*
valueB:@*
dtype0*
_output_shapes
:
e

Reshape_33Reshape	split_2:3Reshape_33/shape*
T0*
Tshape0*
_output_shapes
:@
a
Reshape_34/shapeConst*
valueB"@      *
dtype0*
_output_shapes
:
i

Reshape_34Reshape	split_2:4Reshape_34/shape*
Tshape0*
_output_shapes

:@*
T0
Z
Reshape_35/shapeConst*
valueB:*
dtype0*
_output_shapes
:
e

Reshape_35Reshape	split_2:5Reshape_35/shape*
T0*
Tshape0*
_output_shapes
:
Ђ
beta1_power_1/initial_valueConst*
dtype0*
_output_shapes
: *
valueB
 *fff?*
_class
loc:@v/dense/bias
њ
beta1_power_1
VariableV2*
_output_shapes
: *
shared_name *
_class
loc:@v/dense/bias*
	container *
shape: *
dtype0
х
beta1_power_1/AssignAssignbeta1_power_1beta1_power_1/initial_value*
T0*
_class
loc:@v/dense/bias*
validate_shape(*
_output_shapes
: *
use_locking(
o
beta1_power_1/readIdentitybeta1_power_1*
_output_shapes
: *
T0*
_class
loc:@v/dense/bias
Ђ
beta2_power_1/initial_valueConst*
_output_shapes
: *
valueB
 *wЙ?*
_class
loc:@v/dense/bias*
dtype0
њ
beta2_power_1
VariableV2*
shared_name *
_class
loc:@v/dense/bias*
	container *
shape: *
dtype0*
_output_shapes
: 
х
beta2_power_1/AssignAssignbeta2_power_1beta2_power_1/initial_value*
_output_shapes
: *
use_locking(*
T0*
_class
loc:@v/dense/bias*
validate_shape(
o
beta2_power_1/readIdentitybeta2_power_1*
_output_shapes
: *
T0*
_class
loc:@v/dense/bias
Е
5v/dense/kernel/Adam/Initializer/zeros/shape_as_tensorConst*!
_class
loc:@v/dense/kernel*
valueB"Є   @   *
dtype0*
_output_shapes
:
Њ
+v/dense/kernel/Adam/Initializer/zeros/ConstConst*!
_class
loc:@v/dense/kernel*
valueB
 *    *
dtype0*
_output_shapes
: 
­
%v/dense/kernel/Adam/Initializer/zerosFill5v/dense/kernel/Adam/Initializer/zeros/shape_as_tensor+v/dense/kernel/Adam/Initializer/zeros/Const*
_output_shapes
:	Є@*
T0*!
_class
loc:@v/dense/kernel*

index_type0
г
v/dense/kernel/Adam
VariableV2*
shared_name *!
_class
loc:@v/dense/kernel*
	container *
shape:	Є@*
dtype0*
_output_shapes
:	Є@
о
v/dense/kernel/Adam/AssignAssignv/dense/kernel/Adam%v/dense/kernel/Adam/Initializer/zeros*
T0*!
_class
loc:@v/dense/kernel*
validate_shape(*
_output_shapes
:	Є@*
use_locking(
є
v/dense/kernel/Adam/readIdentityv/dense/kernel/Adam*
T0*!
_class
loc:@v/dense/kernel*
_output_shapes
:	Є@
Ф
7v/dense/kernel/Adam_1/Initializer/zeros/shape_as_tensorConst*!
_class
loc:@v/dense/kernel*
valueB"Є   @   *
dtype0*
_output_shapes
:
Ћ
-v/dense/kernel/Adam_1/Initializer/zeros/ConstConst*!
_class
loc:@v/dense/kernel*
valueB
 *    *
dtype0*
_output_shapes
: 
Ш
'v/dense/kernel/Adam_1/Initializer/zerosFill7v/dense/kernel/Adam_1/Initializer/zeros/shape_as_tensor-v/dense/kernel/Adam_1/Initializer/zeros/Const*
_output_shapes
:	Є@*
T0*!
_class
loc:@v/dense/kernel*

index_type0
«
v/dense/kernel/Adam_1
VariableV2*
dtype0*
_output_shapes
:	Є@*
shared_name *!
_class
loc:@v/dense/kernel*
	container *
shape:	Є@
▄
v/dense/kernel/Adam_1/AssignAssignv/dense/kernel/Adam_1'v/dense/kernel/Adam_1/Initializer/zeros*
use_locking(*
T0*!
_class
loc:@v/dense/kernel*
validate_shape(*
_output_shapes
:	Є@
і
v/dense/kernel/Adam_1/readIdentityv/dense/kernel/Adam_1*
T0*!
_class
loc:@v/dense/kernel*
_output_shapes
:	Є@
Љ
#v/dense/bias/Adam/Initializer/zerosConst*
dtype0*
_output_shapes
:@*
_class
loc:@v/dense/bias*
valueB@*    
ъ
v/dense/bias/Adam
VariableV2*
shared_name *
_class
loc:@v/dense/bias*
	container *
shape:@*
dtype0*
_output_shapes
:@
╔
v/dense/bias/Adam/AssignAssignv/dense/bias/Adam#v/dense/bias/Adam/Initializer/zeros*
use_locking(*
T0*
_class
loc:@v/dense/bias*
validate_shape(*
_output_shapes
:@
{
v/dense/bias/Adam/readIdentityv/dense/bias/Adam*
_class
loc:@v/dense/bias*
_output_shapes
:@*
T0
Њ
%v/dense/bias/Adam_1/Initializer/zerosConst*
_class
loc:@v/dense/bias*
valueB@*    *
dtype0*
_output_shapes
:@
а
v/dense/bias/Adam_1
VariableV2*
_output_shapes
:@*
shared_name *
_class
loc:@v/dense/bias*
	container *
shape:@*
dtype0
¤
v/dense/bias/Adam_1/AssignAssignv/dense/bias/Adam_1%v/dense/bias/Adam_1/Initializer/zeros*
_class
loc:@v/dense/bias*
validate_shape(*
_output_shapes
:@*
use_locking(*
T0

v/dense/bias/Adam_1/readIdentityv/dense/bias/Adam_1*
T0*
_class
loc:@v/dense/bias*
_output_shapes
:@
Г
7v/dense_1/kernel/Adam/Initializer/zeros/shape_as_tensorConst*#
_class
loc:@v/dense_1/kernel*
valueB"@   @   *
dtype0*
_output_shapes
:
Ќ
-v/dense_1/kernel/Adam/Initializer/zeros/ConstConst*#
_class
loc:@v/dense_1/kernel*
valueB
 *    *
dtype0*
_output_shapes
: 
э
'v/dense_1/kernel/Adam/Initializer/zerosFill7v/dense_1/kernel/Adam/Initializer/zeros/shape_as_tensor-v/dense_1/kernel/Adam/Initializer/zeros/Const*
T0*#
_class
loc:@v/dense_1/kernel*

index_type0*
_output_shapes

:@@
«
v/dense_1/kernel/Adam
VariableV2*#
_class
loc:@v/dense_1/kernel*
	container *
shape
:@@*
dtype0*
_output_shapes

:@@*
shared_name 
П
v/dense_1/kernel/Adam/AssignAssignv/dense_1/kernel/Adam'v/dense_1/kernel/Adam/Initializer/zeros*
use_locking(*
T0*#
_class
loc:@v/dense_1/kernel*
validate_shape(*
_output_shapes

:@@
І
v/dense_1/kernel/Adam/readIdentityv/dense_1/kernel/Adam*
T0*#
_class
loc:@v/dense_1/kernel*
_output_shapes

:@@
»
9v/dense_1/kernel/Adam_1/Initializer/zeros/shape_as_tensorConst*#
_class
loc:@v/dense_1/kernel*
valueB"@   @   *
dtype0*
_output_shapes
:
Ў
/v/dense_1/kernel/Adam_1/Initializer/zeros/ConstConst*#
_class
loc:@v/dense_1/kernel*
valueB
 *    *
dtype0*
_output_shapes
: 
§
)v/dense_1/kernel/Adam_1/Initializer/zerosFill9v/dense_1/kernel/Adam_1/Initializer/zeros/shape_as_tensor/v/dense_1/kernel/Adam_1/Initializer/zeros/Const*
T0*#
_class
loc:@v/dense_1/kernel*

index_type0*
_output_shapes

:@@
░
v/dense_1/kernel/Adam_1
VariableV2*
_output_shapes

:@@*
shared_name *#
_class
loc:@v/dense_1/kernel*
	container *
shape
:@@*
dtype0
с
v/dense_1/kernel/Adam_1/AssignAssignv/dense_1/kernel/Adam_1)v/dense_1/kernel/Adam_1/Initializer/zeros*
T0*#
_class
loc:@v/dense_1/kernel*
validate_shape(*
_output_shapes

:@@*
use_locking(
Ј
v/dense_1/kernel/Adam_1/readIdentityv/dense_1/kernel/Adam_1*
_output_shapes

:@@*
T0*#
_class
loc:@v/dense_1/kernel
Ћ
%v/dense_1/bias/Adam/Initializer/zerosConst*!
_class
loc:@v/dense_1/bias*
valueB@*    *
dtype0*
_output_shapes
:@
б
v/dense_1/bias/Adam
VariableV2*
shape:@*
dtype0*
_output_shapes
:@*
shared_name *!
_class
loc:@v/dense_1/bias*
	container 
Л
v/dense_1/bias/Adam/AssignAssignv/dense_1/bias/Adam%v/dense_1/bias/Adam/Initializer/zeros*
validate_shape(*
_output_shapes
:@*
use_locking(*
T0*!
_class
loc:@v/dense_1/bias
Ђ
v/dense_1/bias/Adam/readIdentityv/dense_1/bias/Adam*
_output_shapes
:@*
T0*!
_class
loc:@v/dense_1/bias
Ќ
'v/dense_1/bias/Adam_1/Initializer/zerosConst*!
_class
loc:@v/dense_1/bias*
valueB@*    *
dtype0*
_output_shapes
:@
ц
v/dense_1/bias/Adam_1
VariableV2*!
_class
loc:@v/dense_1/bias*
	container *
shape:@*
dtype0*
_output_shapes
:@*
shared_name 
О
v/dense_1/bias/Adam_1/AssignAssignv/dense_1/bias/Adam_1'v/dense_1/bias/Adam_1/Initializer/zeros*
use_locking(*
T0*!
_class
loc:@v/dense_1/bias*
validate_shape(*
_output_shapes
:@
Ё
v/dense_1/bias/Adam_1/readIdentityv/dense_1/bias/Adam_1*
T0*!
_class
loc:@v/dense_1/bias*
_output_shapes
:@
А
'v/dense_2/kernel/Adam/Initializer/zerosConst*#
_class
loc:@v/dense_2/kernel*
valueB@*    *
dtype0*
_output_shapes

:@
«
v/dense_2/kernel/Adam
VariableV2*
shape
:@*
dtype0*
_output_shapes

:@*
shared_name *#
_class
loc:@v/dense_2/kernel*
	container 
П
v/dense_2/kernel/Adam/AssignAssignv/dense_2/kernel/Adam'v/dense_2/kernel/Adam/Initializer/zeros*
validate_shape(*
_output_shapes

:@*
use_locking(*
T0*#
_class
loc:@v/dense_2/kernel
І
v/dense_2/kernel/Adam/readIdentityv/dense_2/kernel/Adam*
T0*#
_class
loc:@v/dense_2/kernel*
_output_shapes

:@
Б
)v/dense_2/kernel/Adam_1/Initializer/zerosConst*
dtype0*
_output_shapes

:@*#
_class
loc:@v/dense_2/kernel*
valueB@*    
░
v/dense_2/kernel/Adam_1
VariableV2*
dtype0*
_output_shapes

:@*
shared_name *#
_class
loc:@v/dense_2/kernel*
	container *
shape
:@
с
v/dense_2/kernel/Adam_1/AssignAssignv/dense_2/kernel/Adam_1)v/dense_2/kernel/Adam_1/Initializer/zeros*
use_locking(*
T0*#
_class
loc:@v/dense_2/kernel*
validate_shape(*
_output_shapes

:@
Ј
v/dense_2/kernel/Adam_1/readIdentityv/dense_2/kernel/Adam_1*
T0*#
_class
loc:@v/dense_2/kernel*
_output_shapes

:@
Ћ
%v/dense_2/bias/Adam/Initializer/zerosConst*!
_class
loc:@v/dense_2/bias*
valueB*    *
dtype0*
_output_shapes
:
б
v/dense_2/bias/Adam
VariableV2*
_output_shapes
:*
shared_name *!
_class
loc:@v/dense_2/bias*
	container *
shape:*
dtype0
Л
v/dense_2/bias/Adam/AssignAssignv/dense_2/bias/Adam%v/dense_2/bias/Adam/Initializer/zeros*
T0*!
_class
loc:@v/dense_2/bias*
validate_shape(*
_output_shapes
:*
use_locking(
Ђ
v/dense_2/bias/Adam/readIdentityv/dense_2/bias/Adam*
T0*!
_class
loc:@v/dense_2/bias*
_output_shapes
:
Ќ
'v/dense_2/bias/Adam_1/Initializer/zerosConst*!
_class
loc:@v/dense_2/bias*
valueB*    *
dtype0*
_output_shapes
:
ц
v/dense_2/bias/Adam_1
VariableV2*
shape:*
dtype0*
_output_shapes
:*
shared_name *!
_class
loc:@v/dense_2/bias*
	container 
О
v/dense_2/bias/Adam_1/AssignAssignv/dense_2/bias/Adam_1'v/dense_2/bias/Adam_1/Initializer/zeros*
use_locking(*
T0*!
_class
loc:@v/dense_2/bias*
validate_shape(*
_output_shapes
:
Ё
v/dense_2/bias/Adam_1/readIdentityv/dense_2/bias/Adam_1*
_output_shapes
:*
T0*!
_class
loc:@v/dense_2/bias
Y
Adam_1/learning_rateConst*
valueB
 *oЃ:*
dtype0*
_output_shapes
: 
Q
Adam_1/beta1Const*
valueB
 *fff?*
dtype0*
_output_shapes
: 
Q
Adam_1/beta2Const*
valueB
 *wЙ?*
dtype0*
_output_shapes
: 
S
Adam_1/epsilonConst*
valueB
 *w╠+2*
dtype0*
_output_shapes
: 
┘
&Adam_1/update_v/dense/kernel/ApplyAdam	ApplyAdamv/dense/kernelv/dense/kernel/Adamv/dense/kernel/Adam_1beta1_power_1/readbeta2_power_1/readAdam_1/learning_rateAdam_1/beta1Adam_1/beta2Adam_1/epsilon
Reshape_30*
use_locking( *
T0*!
_class
loc:@v/dense/kernel*
use_nesterov( *
_output_shapes
:	Є@
╩
$Adam_1/update_v/dense/bias/ApplyAdam	ApplyAdamv/dense/biasv/dense/bias/Adamv/dense/bias/Adam_1beta1_power_1/readbeta2_power_1/readAdam_1/learning_rateAdam_1/beta1Adam_1/beta2Adam_1/epsilon
Reshape_31*
_output_shapes
:@*
use_locking( *
T0*
_class
loc:@v/dense/bias*
use_nesterov( 
Р
(Adam_1/update_v/dense_1/kernel/ApplyAdam	ApplyAdamv/dense_1/kernelv/dense_1/kernel/Adamv/dense_1/kernel/Adam_1beta1_power_1/readbeta2_power_1/readAdam_1/learning_rateAdam_1/beta1Adam_1/beta2Adam_1/epsilon
Reshape_32*#
_class
loc:@v/dense_1/kernel*
use_nesterov( *
_output_shapes

:@@*
use_locking( *
T0
н
&Adam_1/update_v/dense_1/bias/ApplyAdam	ApplyAdamv/dense_1/biasv/dense_1/bias/Adamv/dense_1/bias/Adam_1beta1_power_1/readbeta2_power_1/readAdam_1/learning_rateAdam_1/beta1Adam_1/beta2Adam_1/epsilon
Reshape_33*
_output_shapes
:@*
use_locking( *
T0*!
_class
loc:@v/dense_1/bias*
use_nesterov( 
Р
(Adam_1/update_v/dense_2/kernel/ApplyAdam	ApplyAdamv/dense_2/kernelv/dense_2/kernel/Adamv/dense_2/kernel/Adam_1beta1_power_1/readbeta2_power_1/readAdam_1/learning_rateAdam_1/beta1Adam_1/beta2Adam_1/epsilon
Reshape_34*
use_locking( *
T0*#
_class
loc:@v/dense_2/kernel*
use_nesterov( *
_output_shapes

:@
н
&Adam_1/update_v/dense_2/bias/ApplyAdam	ApplyAdamv/dense_2/biasv/dense_2/bias/Adamv/dense_2/bias/Adam_1beta1_power_1/readbeta2_power_1/readAdam_1/learning_rateAdam_1/beta1Adam_1/beta2Adam_1/epsilon
Reshape_35*
use_locking( *
T0*!
_class
loc:@v/dense_2/bias*
use_nesterov( *
_output_shapes
:
ь

Adam_1/mulMulbeta1_power_1/readAdam_1/beta1%^Adam_1/update_v/dense/bias/ApplyAdam'^Adam_1/update_v/dense/kernel/ApplyAdam'^Adam_1/update_v/dense_1/bias/ApplyAdam)^Adam_1/update_v/dense_1/kernel/ApplyAdam'^Adam_1/update_v/dense_2/bias/ApplyAdam)^Adam_1/update_v/dense_2/kernel/ApplyAdam*
_class
loc:@v/dense/bias*
_output_shapes
: *
T0
Ю
Adam_1/AssignAssignbeta1_power_1
Adam_1/mul*
_output_shapes
: *
use_locking( *
T0*
_class
loc:@v/dense/bias*
validate_shape(
№
Adam_1/mul_1Mulbeta2_power_1/readAdam_1/beta2%^Adam_1/update_v/dense/bias/ApplyAdam'^Adam_1/update_v/dense/kernel/ApplyAdam'^Adam_1/update_v/dense_1/bias/ApplyAdam)^Adam_1/update_v/dense_1/kernel/ApplyAdam'^Adam_1/update_v/dense_2/bias/ApplyAdam)^Adam_1/update_v/dense_2/kernel/ApplyAdam*
T0*
_class
loc:@v/dense/bias*
_output_shapes
: 
А
Adam_1/Assign_1Assignbeta2_power_1Adam_1/mul_1*
use_locking( *
T0*
_class
loc:@v/dense/bias*
validate_shape(*
_output_shapes
: 
е
Adam_1NoOp^Adam_1/Assign^Adam_1/Assign_1%^Adam_1/update_v/dense/bias/ApplyAdam'^Adam_1/update_v/dense/kernel/ApplyAdam'^Adam_1/update_v/dense_1/bias/ApplyAdam)^Adam_1/update_v/dense_1/kernel/ApplyAdam'^Adam_1/update_v/dense_2/bias/ApplyAdam)^Adam_1/update_v/dense_2/kernel/ApplyAdam
l
Reshape_36/shapeConst^Adam_1*
valueB:
         *
dtype0*
_output_shapes
:
p

Reshape_36Reshapev/dense/kernel/readReshape_36/shape*
T0*
Tshape0*
_output_shapes	
:└C
l
Reshape_37/shapeConst^Adam_1*
valueB:
         *
dtype0*
_output_shapes
:
m

Reshape_37Reshapev/dense/bias/readReshape_37/shape*
T0*
Tshape0*
_output_shapes
:@
l
Reshape_38/shapeConst^Adam_1*
valueB:
         *
dtype0*
_output_shapes
:
r

Reshape_38Reshapev/dense_1/kernel/readReshape_38/shape*
T0*
Tshape0*
_output_shapes	
:ђ 
l
Reshape_39/shapeConst^Adam_1*
valueB:
         *
dtype0*
_output_shapes
:
o

Reshape_39Reshapev/dense_1/bias/readReshape_39/shape*
T0*
Tshape0*
_output_shapes
:@
l
Reshape_40/shapeConst^Adam_1*
valueB:
         *
dtype0*
_output_shapes
:
q

Reshape_40Reshapev/dense_2/kernel/readReshape_40/shape*
Tshape0*
_output_shapes
:@*
T0
l
Reshape_41/shapeConst^Adam_1*
valueB:
         *
dtype0*
_output_shapes
:
o

Reshape_41Reshapev/dense_2/bias/readReshape_41/shape*
_output_shapes
:*
T0*
Tshape0
X
concat_3/axisConst^Adam_1*
dtype0*
_output_shapes
: *
value	B : 
д
concat_3ConcatV2
Reshape_36
Reshape_37
Reshape_38
Reshape_39
Reshape_40
Reshape_41concat_3/axis*
N*
_output_shapes	
:Ђe*

Tidx0*
T0
h
PyFunc_3PyFuncconcat_3*
_output_shapes
:*
Tin
2*
Tout
2*
token
pyfunc_3
q
Const_8Const^Adam_1*-
value$B""└!  @      @   @      *
dtype0*
_output_shapes
:
\
split_3/split_dimConst^Adam_1*
value	B : *
dtype0*
_output_shapes
: 
І
split_3SplitVPyFunc_3Const_8split_3/split_dim*
T0*,
_output_shapes
::::::*
	num_split*

Tlen0
j
Reshape_42/shapeConst^Adam_1*
valueB"Є   @   *
dtype0*
_output_shapes
:
h

Reshape_42Reshapesplit_3Reshape_42/shape*
T0*
Tshape0*
_output_shapes
:	Є@
c
Reshape_43/shapeConst^Adam_1*
valueB:@*
dtype0*
_output_shapes
:
e

Reshape_43Reshape	split_3:1Reshape_43/shape*
T0*
Tshape0*
_output_shapes
:@
j
Reshape_44/shapeConst^Adam_1*
valueB"@   @   *
dtype0*
_output_shapes
:
i

Reshape_44Reshape	split_3:2Reshape_44/shape*
T0*
Tshape0*
_output_shapes

:@@
c
Reshape_45/shapeConst^Adam_1*
dtype0*
_output_shapes
:*
valueB:@
e

Reshape_45Reshape	split_3:3Reshape_45/shape*
_output_shapes
:@*
T0*
Tshape0
j
Reshape_46/shapeConst^Adam_1*
valueB"@      *
dtype0*
_output_shapes
:
i

Reshape_46Reshape	split_3:4Reshape_46/shape*
_output_shapes

:@*
T0*
Tshape0
c
Reshape_47/shapeConst^Adam_1*
valueB:*
dtype0*
_output_shapes
:
e

Reshape_47Reshape	split_3:5Reshape_47/shape*
T0*
Tshape0*
_output_shapes
:
ц
Assign_6Assignv/dense/kernel
Reshape_42*!
_class
loc:@v/dense/kernel*
validate_shape(*
_output_shapes
:	Є@*
use_locking(*
T0
Џ
Assign_7Assignv/dense/bias
Reshape_43*
use_locking(*
T0*
_class
loc:@v/dense/bias*
validate_shape(*
_output_shapes
:@
Д
Assign_8Assignv/dense_1/kernel
Reshape_44*
use_locking(*
T0*#
_class
loc:@v/dense_1/kernel*
validate_shape(*
_output_shapes

:@@
Ъ
Assign_9Assignv/dense_1/bias
Reshape_45*
use_locking(*
T0*!
_class
loc:@v/dense_1/bias*
validate_shape(*
_output_shapes
:@
е
	Assign_10Assignv/dense_2/kernel
Reshape_46*
T0*#
_class
loc:@v/dense_2/kernel*
validate_shape(*
_output_shapes

:@*
use_locking(
а
	Assign_11Assignv/dense_2/bias
Reshape_47*
validate_shape(*
_output_shapes
:*
use_locking(*
T0*!
_class
loc:@v/dense_2/bias
a
group_deps_2NoOp^Adam_1
^Assign_10
^Assign_11	^Assign_6	^Assign_7	^Assign_8	^Assign_9
,
group_deps_3NoOp^Adam_1^group_deps_2
Ы
initNoOp^beta1_power/Assign^beta1_power_1/Assign^beta2_power/Assign^beta2_power_1/Assign^pi/dense/bias/Adam/Assign^pi/dense/bias/Adam_1/Assign^pi/dense/bias/Assign^pi/dense/kernel/Adam/Assign^pi/dense/kernel/Adam_1/Assign^pi/dense/kernel/Assign^pi/dense_1/bias/Adam/Assign^pi/dense_1/bias/Adam_1/Assign^pi/dense_1/bias/Assign^pi/dense_1/kernel/Adam/Assign ^pi/dense_1/kernel/Adam_1/Assign^pi/dense_1/kernel/Assign^pi/dense_2/bias/Adam/Assign^pi/dense_2/bias/Adam_1/Assign^pi/dense_2/bias/Assign^pi/dense_2/kernel/Adam/Assign ^pi/dense_2/kernel/Adam_1/Assign^pi/dense_2/kernel/Assign^v/dense/bias/Adam/Assign^v/dense/bias/Adam_1/Assign^v/dense/bias/Assign^v/dense/kernel/Adam/Assign^v/dense/kernel/Adam_1/Assign^v/dense/kernel/Assign^v/dense_1/bias/Adam/Assign^v/dense_1/bias/Adam_1/Assign^v/dense_1/bias/Assign^v/dense_1/kernel/Adam/Assign^v/dense_1/kernel/Adam_1/Assign^v/dense_1/kernel/Assign^v/dense_2/bias/Adam/Assign^v/dense_2/bias/Adam_1/Assign^v/dense_2/bias/Assign^v/dense_2/kernel/Adam/Assign^v/dense_2/kernel/Adam_1/Assign^v/dense_2/kernel/Assign
c
Reshape_48/shapeConst*
valueB:
         *
dtype0*
_output_shapes
:
q

Reshape_48Reshapepi/dense/kernel/readReshape_48/shape*
T0*
Tshape0*
_output_shapes	
:└C
c
Reshape_49/shapeConst*
valueB:
         *
dtype0*
_output_shapes
:
n

Reshape_49Reshapepi/dense/bias/readReshape_49/shape*
T0*
Tshape0*
_output_shapes
:@
c
Reshape_50/shapeConst*
dtype0*
_output_shapes
:*
valueB:
         
s

Reshape_50Reshapepi/dense_1/kernel/readReshape_50/shape*
T0*
Tshape0*
_output_shapes	
:ђ 
c
Reshape_51/shapeConst*
_output_shapes
:*
valueB:
         *
dtype0
p

Reshape_51Reshapepi/dense_1/bias/readReshape_51/shape*
_output_shapes
:@*
T0*
Tshape0
c
Reshape_52/shapeConst*
valueB:
         *
dtype0*
_output_shapes
:
s

Reshape_52Reshapepi/dense_2/kernel/readReshape_52/shape*
T0*
Tshape0*
_output_shapes	
:ђ
c
Reshape_53/shapeConst*
valueB:
         *
dtype0*
_output_shapes
:
p

Reshape_53Reshapepi/dense_2/bias/readReshape_53/shape*
T0*
Tshape0*
_output_shapes
:
c
Reshape_54/shapeConst*
valueB:
         *
dtype0*
_output_shapes
:
p

Reshape_54Reshapev/dense/kernel/readReshape_54/shape*
T0*
Tshape0*
_output_shapes	
:└C
c
Reshape_55/shapeConst*
valueB:
         *
dtype0*
_output_shapes
:
m

Reshape_55Reshapev/dense/bias/readReshape_55/shape*
T0*
Tshape0*
_output_shapes
:@
c
Reshape_56/shapeConst*
valueB:
         *
dtype0*
_output_shapes
:
r

Reshape_56Reshapev/dense_1/kernel/readReshape_56/shape*
_output_shapes	
:ђ *
T0*
Tshape0
c
Reshape_57/shapeConst*
_output_shapes
:*
valueB:
         *
dtype0
o

Reshape_57Reshapev/dense_1/bias/readReshape_57/shape*
T0*
Tshape0*
_output_shapes
:@
c
Reshape_58/shapeConst*
_output_shapes
:*
valueB:
         *
dtype0
q

Reshape_58Reshapev/dense_2/kernel/readReshape_58/shape*
Tshape0*
_output_shapes
:@*
T0
c
Reshape_59/shapeConst*
valueB:
         *
dtype0*
_output_shapes
:
o

Reshape_59Reshapev/dense_2/bias/readReshape_59/shape*
T0*
Tshape0*
_output_shapes
:
c
Reshape_60/shapeConst*
dtype0*
_output_shapes
:*
valueB:
         
l

Reshape_60Reshapebeta1_power/readReshape_60/shape*
T0*
Tshape0*
_output_shapes
:
c
Reshape_61/shapeConst*
valueB:
         *
dtype0*
_output_shapes
:
l

Reshape_61Reshapebeta2_power/readReshape_61/shape*
T0*
Tshape0*
_output_shapes
:
c
Reshape_62/shapeConst*
valueB:
         *
dtype0*
_output_shapes
:
v

Reshape_62Reshapepi/dense/kernel/Adam/readReshape_62/shape*
T0*
Tshape0*
_output_shapes	
:└C
c
Reshape_63/shapeConst*
dtype0*
_output_shapes
:*
valueB:
         
x

Reshape_63Reshapepi/dense/kernel/Adam_1/readReshape_63/shape*
T0*
Tshape0*
_output_shapes	
:└C
c
Reshape_64/shapeConst*
valueB:
         *
dtype0*
_output_shapes
:
s

Reshape_64Reshapepi/dense/bias/Adam/readReshape_64/shape*
T0*
Tshape0*
_output_shapes
:@
c
Reshape_65/shapeConst*
valueB:
         *
dtype0*
_output_shapes
:
u

Reshape_65Reshapepi/dense/bias/Adam_1/readReshape_65/shape*
T0*
Tshape0*
_output_shapes
:@
c
Reshape_66/shapeConst*
valueB:
         *
dtype0*
_output_shapes
:
x

Reshape_66Reshapepi/dense_1/kernel/Adam/readReshape_66/shape*
T0*
Tshape0*
_output_shapes	
:ђ 
c
Reshape_67/shapeConst*
dtype0*
_output_shapes
:*
valueB:
         
z

Reshape_67Reshapepi/dense_1/kernel/Adam_1/readReshape_67/shape*
_output_shapes	
:ђ *
T0*
Tshape0
c
Reshape_68/shapeConst*
valueB:
         *
dtype0*
_output_shapes
:
u

Reshape_68Reshapepi/dense_1/bias/Adam/readReshape_68/shape*
T0*
Tshape0*
_output_shapes
:@
c
Reshape_69/shapeConst*
valueB:
         *
dtype0*
_output_shapes
:
w

Reshape_69Reshapepi/dense_1/bias/Adam_1/readReshape_69/shape*
T0*
Tshape0*
_output_shapes
:@
c
Reshape_70/shapeConst*
dtype0*
_output_shapes
:*
valueB:
         
x

Reshape_70Reshapepi/dense_2/kernel/Adam/readReshape_70/shape*
T0*
Tshape0*
_output_shapes	
:ђ
c
Reshape_71/shapeConst*
valueB:
         *
dtype0*
_output_shapes
:
z

Reshape_71Reshapepi/dense_2/kernel/Adam_1/readReshape_71/shape*
T0*
Tshape0*
_output_shapes	
:ђ
c
Reshape_72/shapeConst*
_output_shapes
:*
valueB:
         *
dtype0
u

Reshape_72Reshapepi/dense_2/bias/Adam/readReshape_72/shape*
T0*
Tshape0*
_output_shapes
:
c
Reshape_73/shapeConst*
valueB:
         *
dtype0*
_output_shapes
:
w

Reshape_73Reshapepi/dense_2/bias/Adam_1/readReshape_73/shape*
T0*
Tshape0*
_output_shapes
:
c
Reshape_74/shapeConst*
_output_shapes
:*
valueB:
         *
dtype0
n

Reshape_74Reshapebeta1_power_1/readReshape_74/shape*
T0*
Tshape0*
_output_shapes
:
c
Reshape_75/shapeConst*
valueB:
         *
dtype0*
_output_shapes
:
n

Reshape_75Reshapebeta2_power_1/readReshape_75/shape*
Tshape0*
_output_shapes
:*
T0
c
Reshape_76/shapeConst*
valueB:
         *
dtype0*
_output_shapes
:
u

Reshape_76Reshapev/dense/kernel/Adam/readReshape_76/shape*
T0*
Tshape0*
_output_shapes	
:└C
c
Reshape_77/shapeConst*
_output_shapes
:*
valueB:
         *
dtype0
w

Reshape_77Reshapev/dense/kernel/Adam_1/readReshape_77/shape*
T0*
Tshape0*
_output_shapes	
:└C
c
Reshape_78/shapeConst*
valueB:
         *
dtype0*
_output_shapes
:
r

Reshape_78Reshapev/dense/bias/Adam/readReshape_78/shape*
T0*
Tshape0*
_output_shapes
:@
c
Reshape_79/shapeConst*
dtype0*
_output_shapes
:*
valueB:
         
t

Reshape_79Reshapev/dense/bias/Adam_1/readReshape_79/shape*
T0*
Tshape0*
_output_shapes
:@
c
Reshape_80/shapeConst*
valueB:
         *
dtype0*
_output_shapes
:
w

Reshape_80Reshapev/dense_1/kernel/Adam/readReshape_80/shape*
T0*
Tshape0*
_output_shapes	
:ђ 
c
Reshape_81/shapeConst*
valueB:
         *
dtype0*
_output_shapes
:
y

Reshape_81Reshapev/dense_1/kernel/Adam_1/readReshape_81/shape*
T0*
Tshape0*
_output_shapes	
:ђ 
c
Reshape_82/shapeConst*
_output_shapes
:*
valueB:
         *
dtype0
t

Reshape_82Reshapev/dense_1/bias/Adam/readReshape_82/shape*
_output_shapes
:@*
T0*
Tshape0
c
Reshape_83/shapeConst*
_output_shapes
:*
valueB:
         *
dtype0
v

Reshape_83Reshapev/dense_1/bias/Adam_1/readReshape_83/shape*
Tshape0*
_output_shapes
:@*
T0
c
Reshape_84/shapeConst*
dtype0*
_output_shapes
:*
valueB:
         
v

Reshape_84Reshapev/dense_2/kernel/Adam/readReshape_84/shape*
T0*
Tshape0*
_output_shapes
:@
c
Reshape_85/shapeConst*
valueB:
         *
dtype0*
_output_shapes
:
x

Reshape_85Reshapev/dense_2/kernel/Adam_1/readReshape_85/shape*
T0*
Tshape0*
_output_shapes
:@
c
Reshape_86/shapeConst*
valueB:
         *
dtype0*
_output_shapes
:
t

Reshape_86Reshapev/dense_2/bias/Adam/readReshape_86/shape*
_output_shapes
:*
T0*
Tshape0
c
Reshape_87/shapeConst*
valueB:
         *
dtype0*
_output_shapes
:
v

Reshape_87Reshapev/dense_2/bias/Adam_1/readReshape_87/shape*
T0*
Tshape0*
_output_shapes
:
O
concat_4/axisConst*
value	B : *
dtype0*
_output_shapes
: 
┐
concat_4ConcatV2
Reshape_48
Reshape_49
Reshape_50
Reshape_51
Reshape_52
Reshape_53
Reshape_54
Reshape_55
Reshape_56
Reshape_57
Reshape_58
Reshape_59
Reshape_60
Reshape_61
Reshape_62
Reshape_63
Reshape_64
Reshape_65
Reshape_66
Reshape_67
Reshape_68
Reshape_69
Reshape_70
Reshape_71
Reshape_72
Reshape_73
Reshape_74
Reshape_75
Reshape_76
Reshape_77
Reshape_78
Reshape_79
Reshape_80
Reshape_81
Reshape_82
Reshape_83
Reshape_84
Reshape_85
Reshape_86
Reshape_87concat_4/axis*
N(*
_output_shapes

:МР*

Tidx0*
T0
h
PyFunc_4PyFuncconcat_4*
_output_shapes
:*
Tin
2*
Tout
2*
token
pyfunc_4
З
Const_9Const*
dtype0*
_output_shapes
:(*И
value«BФ("а└!  @      @         └!  @      @   @            └!  └!  @   @         @   @                     └!  └!  @   @         @   @   @   @         
S
split_4/split_dimConst*
value	B : *
dtype0*
_output_shapes
: 
ќ
split_4SplitVPyFunc_4Const_9split_4/split_dim*
T0*Х
_output_shapesБ
а::::::::::::::::::::::::::::::::::::::::*
	num_split(*

Tlen0
a
Reshape_88/shapeConst*
valueB"Є   @   *
dtype0*
_output_shapes
:
h

Reshape_88Reshapesplit_4Reshape_88/shape*
T0*
Tshape0*
_output_shapes
:	Є@
Z
Reshape_89/shapeConst*
valueB:@*
dtype0*
_output_shapes
:
e

Reshape_89Reshape	split_4:1Reshape_89/shape*
T0*
Tshape0*
_output_shapes
:@
a
Reshape_90/shapeConst*
valueB"@   @   *
dtype0*
_output_shapes
:
i

Reshape_90Reshape	split_4:2Reshape_90/shape*
T0*
Tshape0*
_output_shapes

:@@
Z
Reshape_91/shapeConst*
valueB:@*
dtype0*
_output_shapes
:
e

Reshape_91Reshape	split_4:3Reshape_91/shape*
T0*
Tshape0*
_output_shapes
:@
a
Reshape_92/shapeConst*
valueB"@      *
dtype0*
_output_shapes
:
i

Reshape_92Reshape	split_4:4Reshape_92/shape*
_output_shapes

:@*
T0*
Tshape0
Z
Reshape_93/shapeConst*
valueB:*
dtype0*
_output_shapes
:
e

Reshape_93Reshape	split_4:5Reshape_93/shape*
T0*
Tshape0*
_output_shapes
:
a
Reshape_94/shapeConst*
valueB"Є   @   *
dtype0*
_output_shapes
:
j

Reshape_94Reshape	split_4:6Reshape_94/shape*
_output_shapes
:	Є@*
T0*
Tshape0
Z
Reshape_95/shapeConst*
valueB:@*
dtype0*
_output_shapes
:
e

Reshape_95Reshape	split_4:7Reshape_95/shape*
T0*
Tshape0*
_output_shapes
:@
a
Reshape_96/shapeConst*
valueB"@   @   *
dtype0*
_output_shapes
:
i

Reshape_96Reshape	split_4:8Reshape_96/shape*
_output_shapes

:@@*
T0*
Tshape0
Z
Reshape_97/shapeConst*
valueB:@*
dtype0*
_output_shapes
:
e

Reshape_97Reshape	split_4:9Reshape_97/shape*
_output_shapes
:@*
T0*
Tshape0
a
Reshape_98/shapeConst*
valueB"@      *
dtype0*
_output_shapes
:
j

Reshape_98Reshape
split_4:10Reshape_98/shape*
T0*
Tshape0*
_output_shapes

:@
Z
Reshape_99/shapeConst*
_output_shapes
:*
valueB:*
dtype0
f

Reshape_99Reshape
split_4:11Reshape_99/shape*
_output_shapes
:*
T0*
Tshape0
T
Reshape_100/shapeConst*
valueB *
dtype0*
_output_shapes
: 
d
Reshape_100Reshape
split_4:12Reshape_100/shape*
_output_shapes
: *
T0*
Tshape0
T
Reshape_101/shapeConst*
valueB *
dtype0*
_output_shapes
: 
d
Reshape_101Reshape
split_4:13Reshape_101/shape*
_output_shapes
: *
T0*
Tshape0
b
Reshape_102/shapeConst*
valueB"Є   @   *
dtype0*
_output_shapes
:
m
Reshape_102Reshape
split_4:14Reshape_102/shape*
T0*
Tshape0*
_output_shapes
:	Є@
b
Reshape_103/shapeConst*
valueB"Є   @   *
dtype0*
_output_shapes
:
m
Reshape_103Reshape
split_4:15Reshape_103/shape*
T0*
Tshape0*
_output_shapes
:	Є@
[
Reshape_104/shapeConst*
valueB:@*
dtype0*
_output_shapes
:
h
Reshape_104Reshape
split_4:16Reshape_104/shape*
_output_shapes
:@*
T0*
Tshape0
[
Reshape_105/shapeConst*
valueB:@*
dtype0*
_output_shapes
:
h
Reshape_105Reshape
split_4:17Reshape_105/shape*
Tshape0*
_output_shapes
:@*
T0
b
Reshape_106/shapeConst*
dtype0*
_output_shapes
:*
valueB"@   @   
l
Reshape_106Reshape
split_4:18Reshape_106/shape*
T0*
Tshape0*
_output_shapes

:@@
b
Reshape_107/shapeConst*
valueB"@   @   *
dtype0*
_output_shapes
:
l
Reshape_107Reshape
split_4:19Reshape_107/shape*
T0*
Tshape0*
_output_shapes

:@@
[
Reshape_108/shapeConst*
valueB:@*
dtype0*
_output_shapes
:
h
Reshape_108Reshape
split_4:20Reshape_108/shape*
T0*
Tshape0*
_output_shapes
:@
[
Reshape_109/shapeConst*
dtype0*
_output_shapes
:*
valueB:@
h
Reshape_109Reshape
split_4:21Reshape_109/shape*
T0*
Tshape0*
_output_shapes
:@
b
Reshape_110/shapeConst*
valueB"@      *
dtype0*
_output_shapes
:
l
Reshape_110Reshape
split_4:22Reshape_110/shape*
T0*
Tshape0*
_output_shapes

:@
b
Reshape_111/shapeConst*
valueB"@      *
dtype0*
_output_shapes
:
l
Reshape_111Reshape
split_4:23Reshape_111/shape*
Tshape0*
_output_shapes

:@*
T0
[
Reshape_112/shapeConst*
valueB:*
dtype0*
_output_shapes
:
h
Reshape_112Reshape
split_4:24Reshape_112/shape*
T0*
Tshape0*
_output_shapes
:
[
Reshape_113/shapeConst*
valueB:*
dtype0*
_output_shapes
:
h
Reshape_113Reshape
split_4:25Reshape_113/shape*
T0*
Tshape0*
_output_shapes
:
T
Reshape_114/shapeConst*
valueB *
dtype0*
_output_shapes
: 
d
Reshape_114Reshape
split_4:26Reshape_114/shape*
_output_shapes
: *
T0*
Tshape0
T
Reshape_115/shapeConst*
valueB *
dtype0*
_output_shapes
: 
d
Reshape_115Reshape
split_4:27Reshape_115/shape*
T0*
Tshape0*
_output_shapes
: 
b
Reshape_116/shapeConst*
valueB"Є   @   *
dtype0*
_output_shapes
:
m
Reshape_116Reshape
split_4:28Reshape_116/shape*
T0*
Tshape0*
_output_shapes
:	Є@
b
Reshape_117/shapeConst*
valueB"Є   @   *
dtype0*
_output_shapes
:
m
Reshape_117Reshape
split_4:29Reshape_117/shape*
T0*
Tshape0*
_output_shapes
:	Є@
[
Reshape_118/shapeConst*
valueB:@*
dtype0*
_output_shapes
:
h
Reshape_118Reshape
split_4:30Reshape_118/shape*
_output_shapes
:@*
T0*
Tshape0
[
Reshape_119/shapeConst*
dtype0*
_output_shapes
:*
valueB:@
h
Reshape_119Reshape
split_4:31Reshape_119/shape*
T0*
Tshape0*
_output_shapes
:@
b
Reshape_120/shapeConst*
valueB"@   @   *
dtype0*
_output_shapes
:
l
Reshape_120Reshape
split_4:32Reshape_120/shape*
Tshape0*
_output_shapes

:@@*
T0
b
Reshape_121/shapeConst*
valueB"@   @   *
dtype0*
_output_shapes
:
l
Reshape_121Reshape
split_4:33Reshape_121/shape*
T0*
Tshape0*
_output_shapes

:@@
[
Reshape_122/shapeConst*
valueB:@*
dtype0*
_output_shapes
:
h
Reshape_122Reshape
split_4:34Reshape_122/shape*
Tshape0*
_output_shapes
:@*
T0
[
Reshape_123/shapeConst*
valueB:@*
dtype0*
_output_shapes
:
h
Reshape_123Reshape
split_4:35Reshape_123/shape*
_output_shapes
:@*
T0*
Tshape0
b
Reshape_124/shapeConst*
valueB"@      *
dtype0*
_output_shapes
:
l
Reshape_124Reshape
split_4:36Reshape_124/shape*
T0*
Tshape0*
_output_shapes

:@
b
Reshape_125/shapeConst*
valueB"@      *
dtype0*
_output_shapes
:
l
Reshape_125Reshape
split_4:37Reshape_125/shape*
T0*
Tshape0*
_output_shapes

:@
[
Reshape_126/shapeConst*
valueB:*
dtype0*
_output_shapes
:
h
Reshape_126Reshape
split_4:38Reshape_126/shape*
T0*
Tshape0*
_output_shapes
:
[
Reshape_127/shapeConst*
valueB:*
dtype0*
_output_shapes
:
h
Reshape_127Reshape
split_4:39Reshape_127/shape*
_output_shapes
:*
T0*
Tshape0
Д
	Assign_12Assignpi/dense/kernel
Reshape_88*
_output_shapes
:	Є@*
use_locking(*
T0*"
_class
loc:@pi/dense/kernel*
validate_shape(
ъ
	Assign_13Assignpi/dense/bias
Reshape_89*
use_locking(*
T0* 
_class
loc:@pi/dense/bias*
validate_shape(*
_output_shapes
:@
ф
	Assign_14Assignpi/dense_1/kernel
Reshape_90*$
_class
loc:@pi/dense_1/kernel*
validate_shape(*
_output_shapes

:@@*
use_locking(*
T0
б
	Assign_15Assignpi/dense_1/bias
Reshape_91*
use_locking(*
T0*"
_class
loc:@pi/dense_1/bias*
validate_shape(*
_output_shapes
:@
ф
	Assign_16Assignpi/dense_2/kernel
Reshape_92*
use_locking(*
T0*$
_class
loc:@pi/dense_2/kernel*
validate_shape(*
_output_shapes

:@
б
	Assign_17Assignpi/dense_2/bias
Reshape_93*
use_locking(*
T0*"
_class
loc:@pi/dense_2/bias*
validate_shape(*
_output_shapes
:
Ц
	Assign_18Assignv/dense/kernel
Reshape_94*!
_class
loc:@v/dense/kernel*
validate_shape(*
_output_shapes
:	Є@*
use_locking(*
T0
ю
	Assign_19Assignv/dense/bias
Reshape_95*
use_locking(*
T0*
_class
loc:@v/dense/bias*
validate_shape(*
_output_shapes
:@
е
	Assign_20Assignv/dense_1/kernel
Reshape_96*
use_locking(*
T0*#
_class
loc:@v/dense_1/kernel*
validate_shape(*
_output_shapes

:@@
а
	Assign_21Assignv/dense_1/bias
Reshape_97*
use_locking(*
T0*!
_class
loc:@v/dense_1/bias*
validate_shape(*
_output_shapes
:@
е
	Assign_22Assignv/dense_2/kernel
Reshape_98*
use_locking(*
T0*#
_class
loc:@v/dense_2/kernel*
validate_shape(*
_output_shapes

:@
а
	Assign_23Assignv/dense_2/bias
Reshape_99*
validate_shape(*
_output_shapes
:*
use_locking(*
T0*!
_class
loc:@v/dense_2/bias
Ў
	Assign_24Assignbeta1_powerReshape_100* 
_class
loc:@pi/dense/bias*
validate_shape(*
_output_shapes
: *
use_locking(*
T0
Ў
	Assign_25Assignbeta2_powerReshape_101*
T0* 
_class
loc:@pi/dense/bias*
validate_shape(*
_output_shapes
: *
use_locking(
Г
	Assign_26Assignpi/dense/kernel/AdamReshape_102*
use_locking(*
T0*"
_class
loc:@pi/dense/kernel*
validate_shape(*
_output_shapes
:	Є@
»
	Assign_27Assignpi/dense/kernel/Adam_1Reshape_103*
_output_shapes
:	Є@*
use_locking(*
T0*"
_class
loc:@pi/dense/kernel*
validate_shape(
ц
	Assign_28Assignpi/dense/bias/AdamReshape_104*
validate_shape(*
_output_shapes
:@*
use_locking(*
T0* 
_class
loc:@pi/dense/bias
д
	Assign_29Assignpi/dense/bias/Adam_1Reshape_105* 
_class
loc:@pi/dense/bias*
validate_shape(*
_output_shapes
:@*
use_locking(*
T0
░
	Assign_30Assignpi/dense_1/kernel/AdamReshape_106*
use_locking(*
T0*$
_class
loc:@pi/dense_1/kernel*
validate_shape(*
_output_shapes

:@@
▓
	Assign_31Assignpi/dense_1/kernel/Adam_1Reshape_107*
validate_shape(*
_output_shapes

:@@*
use_locking(*
T0*$
_class
loc:@pi/dense_1/kernel
е
	Assign_32Assignpi/dense_1/bias/AdamReshape_108*
validate_shape(*
_output_shapes
:@*
use_locking(*
T0*"
_class
loc:@pi/dense_1/bias
ф
	Assign_33Assignpi/dense_1/bias/Adam_1Reshape_109*
use_locking(*
T0*"
_class
loc:@pi/dense_1/bias*
validate_shape(*
_output_shapes
:@
░
	Assign_34Assignpi/dense_2/kernel/AdamReshape_110*$
_class
loc:@pi/dense_2/kernel*
validate_shape(*
_output_shapes

:@*
use_locking(*
T0
▓
	Assign_35Assignpi/dense_2/kernel/Adam_1Reshape_111*
use_locking(*
T0*$
_class
loc:@pi/dense_2/kernel*
validate_shape(*
_output_shapes

:@
е
	Assign_36Assignpi/dense_2/bias/AdamReshape_112*
use_locking(*
T0*"
_class
loc:@pi/dense_2/bias*
validate_shape(*
_output_shapes
:
ф
	Assign_37Assignpi/dense_2/bias/Adam_1Reshape_113*
T0*"
_class
loc:@pi/dense_2/bias*
validate_shape(*
_output_shapes
:*
use_locking(
џ
	Assign_38Assignbeta1_power_1Reshape_114*
validate_shape(*
_output_shapes
: *
use_locking(*
T0*
_class
loc:@v/dense/bias
џ
	Assign_39Assignbeta2_power_1Reshape_115*
validate_shape(*
_output_shapes
: *
use_locking(*
T0*
_class
loc:@v/dense/bias
Ф
	Assign_40Assignv/dense/kernel/AdamReshape_116*
use_locking(*
T0*!
_class
loc:@v/dense/kernel*
validate_shape(*
_output_shapes
:	Є@
Г
	Assign_41Assignv/dense/kernel/Adam_1Reshape_117*!
_class
loc:@v/dense/kernel*
validate_shape(*
_output_shapes
:	Є@*
use_locking(*
T0
б
	Assign_42Assignv/dense/bias/AdamReshape_118*
use_locking(*
T0*
_class
loc:@v/dense/bias*
validate_shape(*
_output_shapes
:@
ц
	Assign_43Assignv/dense/bias/Adam_1Reshape_119*
T0*
_class
loc:@v/dense/bias*
validate_shape(*
_output_shapes
:@*
use_locking(
«
	Assign_44Assignv/dense_1/kernel/AdamReshape_120*
use_locking(*
T0*#
_class
loc:@v/dense_1/kernel*
validate_shape(*
_output_shapes

:@@
░
	Assign_45Assignv/dense_1/kernel/Adam_1Reshape_121*
T0*#
_class
loc:@v/dense_1/kernel*
validate_shape(*
_output_shapes

:@@*
use_locking(
д
	Assign_46Assignv/dense_1/bias/AdamReshape_122*
validate_shape(*
_output_shapes
:@*
use_locking(*
T0*!
_class
loc:@v/dense_1/bias
е
	Assign_47Assignv/dense_1/bias/Adam_1Reshape_123*
use_locking(*
T0*!
_class
loc:@v/dense_1/bias*
validate_shape(*
_output_shapes
:@
«
	Assign_48Assignv/dense_2/kernel/AdamReshape_124*
use_locking(*
T0*#
_class
loc:@v/dense_2/kernel*
validate_shape(*
_output_shapes

:@
░
	Assign_49Assignv/dense_2/kernel/Adam_1Reshape_125*
validate_shape(*
_output_shapes

:@*
use_locking(*
T0*#
_class
loc:@v/dense_2/kernel
д
	Assign_50Assignv/dense_2/bias/AdamReshape_126*
_output_shapes
:*
use_locking(*
T0*!
_class
loc:@v/dense_2/bias*
validate_shape(
е
	Assign_51Assignv/dense_2/bias/Adam_1Reshape_127*
_output_shapes
:*
use_locking(*
T0*!
_class
loc:@v/dense_2/bias*
validate_shape(
З
group_deps_4NoOp
^Assign_12
^Assign_13
^Assign_14
^Assign_15
^Assign_16
^Assign_17
^Assign_18
^Assign_19
^Assign_20
^Assign_21
^Assign_22
^Assign_23
^Assign_24
^Assign_25
^Assign_26
^Assign_27
^Assign_28
^Assign_29
^Assign_30
^Assign_31
^Assign_32
^Assign_33
^Assign_34
^Assign_35
^Assign_36
^Assign_37
^Assign_38
^Assign_39
^Assign_40
^Assign_41
^Assign_42
^Assign_43
^Assign_44
^Assign_45
^Assign_46
^Assign_47
^Assign_48
^Assign_49
^Assign_50
^Assign_51
Y
save/filename/inputConst*
valueB Bmodel*
dtype0*
_output_shapes
: 
n
save/filenamePlaceholderWithDefaultsave/filename/input*
dtype0*
_output_shapes
: *
shape: 
e

save/ConstPlaceholderWithDefaultsave/filename*
dtype0*
_output_shapes
: *
shape: 
ё
save/StringJoin/inputs_1Const*<
value3B1 B+_temp_7f02eafcc18a4176a7cfda9c6405db86/part*
dtype0*
_output_shapes
: 
u
save/StringJoin
StringJoin
save/Constsave/StringJoin/inputs_1*
N*
_output_shapes
: *
	separator 
Q
save/num_shardsConst*
value	B :*
dtype0*
_output_shapes
: 
\
save/ShardedFilename/shardConst*
value	B : *
dtype0*
_output_shapes
: 
}
save/ShardedFilenameShardedFilenamesave/StringJoinsave/ShardedFilename/shardsave/num_shards*
_output_shapes
: 
ѕ
save/SaveV2/tensor_namesConst*╗
value▒B«(Bbeta1_powerBbeta1_power_1Bbeta2_powerBbeta2_power_1Bpi/dense/biasBpi/dense/bias/AdamBpi/dense/bias/Adam_1Bpi/dense/kernelBpi/dense/kernel/AdamBpi/dense/kernel/Adam_1Bpi/dense_1/biasBpi/dense_1/bias/AdamBpi/dense_1/bias/Adam_1Bpi/dense_1/kernelBpi/dense_1/kernel/AdamBpi/dense_1/kernel/Adam_1Bpi/dense_2/biasBpi/dense_2/bias/AdamBpi/dense_2/bias/Adam_1Bpi/dense_2/kernelBpi/dense_2/kernel/AdamBpi/dense_2/kernel/Adam_1Bv/dense/biasBv/dense/bias/AdamBv/dense/bias/Adam_1Bv/dense/kernelBv/dense/kernel/AdamBv/dense/kernel/Adam_1Bv/dense_1/biasBv/dense_1/bias/AdamBv/dense_1/bias/Adam_1Bv/dense_1/kernelBv/dense_1/kernel/AdamBv/dense_1/kernel/Adam_1Bv/dense_2/biasBv/dense_2/bias/AdamBv/dense_2/bias/Adam_1Bv/dense_2/kernelBv/dense_2/kernel/AdamBv/dense_2/kernel/Adam_1*
dtype0*
_output_shapes
:(
│
save/SaveV2/shape_and_slicesConst*c
valueZBX(B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B *
dtype0*
_output_shapes
:(
┴
save/SaveV2SaveV2save/ShardedFilenamesave/SaveV2/tensor_namessave/SaveV2/shape_and_slicesbeta1_powerbeta1_power_1beta2_powerbeta2_power_1pi/dense/biaspi/dense/bias/Adampi/dense/bias/Adam_1pi/dense/kernelpi/dense/kernel/Adampi/dense/kernel/Adam_1pi/dense_1/biaspi/dense_1/bias/Adampi/dense_1/bias/Adam_1pi/dense_1/kernelpi/dense_1/kernel/Adampi/dense_1/kernel/Adam_1pi/dense_2/biaspi/dense_2/bias/Adampi/dense_2/bias/Adam_1pi/dense_2/kernelpi/dense_2/kernel/Adampi/dense_2/kernel/Adam_1v/dense/biasv/dense/bias/Adamv/dense/bias/Adam_1v/dense/kernelv/dense/kernel/Adamv/dense/kernel/Adam_1v/dense_1/biasv/dense_1/bias/Adamv/dense_1/bias/Adam_1v/dense_1/kernelv/dense_1/kernel/Adamv/dense_1/kernel/Adam_1v/dense_2/biasv/dense_2/bias/Adamv/dense_2/bias/Adam_1v/dense_2/kernelv/dense_2/kernel/Adamv/dense_2/kernel/Adam_1*6
dtypes,
*2(
Љ
save/control_dependencyIdentitysave/ShardedFilename^save/SaveV2*
_output_shapes
: *
T0*'
_class
loc:@save/ShardedFilename
Ю
+save/MergeV2Checkpoints/checkpoint_prefixesPacksave/ShardedFilename^save/control_dependency*
_output_shapes
:*
T0*

axis *
N
}
save/MergeV2CheckpointsMergeV2Checkpoints+save/MergeV2Checkpoints/checkpoint_prefixes
save/Const*
delete_old_dirs(
z
save/IdentityIdentity
save/Const^save/MergeV2Checkpoints^save/control_dependency*
T0*
_output_shapes
: 
І
save/RestoreV2/tensor_namesConst*╗
value▒B«(Bbeta1_powerBbeta1_power_1Bbeta2_powerBbeta2_power_1Bpi/dense/biasBpi/dense/bias/AdamBpi/dense/bias/Adam_1Bpi/dense/kernelBpi/dense/kernel/AdamBpi/dense/kernel/Adam_1Bpi/dense_1/biasBpi/dense_1/bias/AdamBpi/dense_1/bias/Adam_1Bpi/dense_1/kernelBpi/dense_1/kernel/AdamBpi/dense_1/kernel/Adam_1Bpi/dense_2/biasBpi/dense_2/bias/AdamBpi/dense_2/bias/Adam_1Bpi/dense_2/kernelBpi/dense_2/kernel/AdamBpi/dense_2/kernel/Adam_1Bv/dense/biasBv/dense/bias/AdamBv/dense/bias/Adam_1Bv/dense/kernelBv/dense/kernel/AdamBv/dense/kernel/Adam_1Bv/dense_1/biasBv/dense_1/bias/AdamBv/dense_1/bias/Adam_1Bv/dense_1/kernelBv/dense_1/kernel/AdamBv/dense_1/kernel/Adam_1Bv/dense_2/biasBv/dense_2/bias/AdamBv/dense_2/bias/Adam_1Bv/dense_2/kernelBv/dense_2/kernel/AdamBv/dense_2/kernel/Adam_1*
dtype0*
_output_shapes
:(
Х
save/RestoreV2/shape_and_slicesConst*
dtype0*
_output_shapes
:(*c
valueZBX(B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 
о
save/RestoreV2	RestoreV2
save/Constsave/RestoreV2/tensor_namessave/RestoreV2/shape_and_slices*6
dtypes,
*2(*Х
_output_shapesБ
а::::::::::::::::::::::::::::::::::::::::
ъ
save/AssignAssignbeta1_powersave/RestoreV2*
use_locking(*
T0* 
_class
loc:@pi/dense/bias*
validate_shape(*
_output_shapes
: 
Б
save/Assign_1Assignbeta1_power_1save/RestoreV2:1*
_class
loc:@v/dense/bias*
validate_shape(*
_output_shapes
: *
use_locking(*
T0
б
save/Assign_2Assignbeta2_powersave/RestoreV2:2*
use_locking(*
T0* 
_class
loc:@pi/dense/bias*
validate_shape(*
_output_shapes
: 
Б
save/Assign_3Assignbeta2_power_1save/RestoreV2:3*
_output_shapes
: *
use_locking(*
T0*
_class
loc:@v/dense/bias*
validate_shape(
е
save/Assign_4Assignpi/dense/biassave/RestoreV2:4*
validate_shape(*
_output_shapes
:@*
use_locking(*
T0* 
_class
loc:@pi/dense/bias
Г
save/Assign_5Assignpi/dense/bias/Adamsave/RestoreV2:5*
T0* 
_class
loc:@pi/dense/bias*
validate_shape(*
_output_shapes
:@*
use_locking(
»
save/Assign_6Assignpi/dense/bias/Adam_1save/RestoreV2:6*
use_locking(*
T0* 
_class
loc:@pi/dense/bias*
validate_shape(*
_output_shapes
:@
▒
save/Assign_7Assignpi/dense/kernelsave/RestoreV2:7*
validate_shape(*
_output_shapes
:	Є@*
use_locking(*
T0*"
_class
loc:@pi/dense/kernel
Х
save/Assign_8Assignpi/dense/kernel/Adamsave/RestoreV2:8*
use_locking(*
T0*"
_class
loc:@pi/dense/kernel*
validate_shape(*
_output_shapes
:	Є@
И
save/Assign_9Assignpi/dense/kernel/Adam_1save/RestoreV2:9*"
_class
loc:@pi/dense/kernel*
validate_shape(*
_output_shapes
:	Є@*
use_locking(*
T0
«
save/Assign_10Assignpi/dense_1/biassave/RestoreV2:10*
use_locking(*
T0*"
_class
loc:@pi/dense_1/bias*
validate_shape(*
_output_shapes
:@
│
save/Assign_11Assignpi/dense_1/bias/Adamsave/RestoreV2:11*
use_locking(*
T0*"
_class
loc:@pi/dense_1/bias*
validate_shape(*
_output_shapes
:@
х
save/Assign_12Assignpi/dense_1/bias/Adam_1save/RestoreV2:12*"
_class
loc:@pi/dense_1/bias*
validate_shape(*
_output_shapes
:@*
use_locking(*
T0
Х
save/Assign_13Assignpi/dense_1/kernelsave/RestoreV2:13*
use_locking(*
T0*$
_class
loc:@pi/dense_1/kernel*
validate_shape(*
_output_shapes

:@@
╗
save/Assign_14Assignpi/dense_1/kernel/Adamsave/RestoreV2:14*
use_locking(*
T0*$
_class
loc:@pi/dense_1/kernel*
validate_shape(*
_output_shapes

:@@
й
save/Assign_15Assignpi/dense_1/kernel/Adam_1save/RestoreV2:15*
use_locking(*
T0*$
_class
loc:@pi/dense_1/kernel*
validate_shape(*
_output_shapes

:@@
«
save/Assign_16Assignpi/dense_2/biassave/RestoreV2:16*
_output_shapes
:*
use_locking(*
T0*"
_class
loc:@pi/dense_2/bias*
validate_shape(
│
save/Assign_17Assignpi/dense_2/bias/Adamsave/RestoreV2:17*"
_class
loc:@pi/dense_2/bias*
validate_shape(*
_output_shapes
:*
use_locking(*
T0
х
save/Assign_18Assignpi/dense_2/bias/Adam_1save/RestoreV2:18*
validate_shape(*
_output_shapes
:*
use_locking(*
T0*"
_class
loc:@pi/dense_2/bias
Х
save/Assign_19Assignpi/dense_2/kernelsave/RestoreV2:19*
T0*$
_class
loc:@pi/dense_2/kernel*
validate_shape(*
_output_shapes

:@*
use_locking(
╗
save/Assign_20Assignpi/dense_2/kernel/Adamsave/RestoreV2:20*
use_locking(*
T0*$
_class
loc:@pi/dense_2/kernel*
validate_shape(*
_output_shapes

:@
й
save/Assign_21Assignpi/dense_2/kernel/Adam_1save/RestoreV2:21*
use_locking(*
T0*$
_class
loc:@pi/dense_2/kernel*
validate_shape(*
_output_shapes

:@
е
save/Assign_22Assignv/dense/biassave/RestoreV2:22*
validate_shape(*
_output_shapes
:@*
use_locking(*
T0*
_class
loc:@v/dense/bias
Г
save/Assign_23Assignv/dense/bias/Adamsave/RestoreV2:23*
use_locking(*
T0*
_class
loc:@v/dense/bias*
validate_shape(*
_output_shapes
:@
»
save/Assign_24Assignv/dense/bias/Adam_1save/RestoreV2:24*
_class
loc:@v/dense/bias*
validate_shape(*
_output_shapes
:@*
use_locking(*
T0
▒
save/Assign_25Assignv/dense/kernelsave/RestoreV2:25*
validate_shape(*
_output_shapes
:	Є@*
use_locking(*
T0*!
_class
loc:@v/dense/kernel
Х
save/Assign_26Assignv/dense/kernel/Adamsave/RestoreV2:26*
use_locking(*
T0*!
_class
loc:@v/dense/kernel*
validate_shape(*
_output_shapes
:	Є@
И
save/Assign_27Assignv/dense/kernel/Adam_1save/RestoreV2:27*
use_locking(*
T0*!
_class
loc:@v/dense/kernel*
validate_shape(*
_output_shapes
:	Є@
г
save/Assign_28Assignv/dense_1/biassave/RestoreV2:28*
use_locking(*
T0*!
_class
loc:@v/dense_1/bias*
validate_shape(*
_output_shapes
:@
▒
save/Assign_29Assignv/dense_1/bias/Adamsave/RestoreV2:29*
use_locking(*
T0*!
_class
loc:@v/dense_1/bias*
validate_shape(*
_output_shapes
:@
│
save/Assign_30Assignv/dense_1/bias/Adam_1save/RestoreV2:30*
T0*!
_class
loc:@v/dense_1/bias*
validate_shape(*
_output_shapes
:@*
use_locking(
┤
save/Assign_31Assignv/dense_1/kernelsave/RestoreV2:31*
use_locking(*
T0*#
_class
loc:@v/dense_1/kernel*
validate_shape(*
_output_shapes

:@@
╣
save/Assign_32Assignv/dense_1/kernel/Adamsave/RestoreV2:32*
validate_shape(*
_output_shapes

:@@*
use_locking(*
T0*#
_class
loc:@v/dense_1/kernel
╗
save/Assign_33Assignv/dense_1/kernel/Adam_1save/RestoreV2:33*
T0*#
_class
loc:@v/dense_1/kernel*
validate_shape(*
_output_shapes

:@@*
use_locking(
г
save/Assign_34Assignv/dense_2/biassave/RestoreV2:34*
use_locking(*
T0*!
_class
loc:@v/dense_2/bias*
validate_shape(*
_output_shapes
:
▒
save/Assign_35Assignv/dense_2/bias/Adamsave/RestoreV2:35*
use_locking(*
T0*!
_class
loc:@v/dense_2/bias*
validate_shape(*
_output_shapes
:
│
save/Assign_36Assignv/dense_2/bias/Adam_1save/RestoreV2:36*
use_locking(*
T0*!
_class
loc:@v/dense_2/bias*
validate_shape(*
_output_shapes
:
┤
save/Assign_37Assignv/dense_2/kernelsave/RestoreV2:37*
_output_shapes

:@*
use_locking(*
T0*#
_class
loc:@v/dense_2/kernel*
validate_shape(
╣
save/Assign_38Assignv/dense_2/kernel/Adamsave/RestoreV2:38*
use_locking(*
T0*#
_class
loc:@v/dense_2/kernel*
validate_shape(*
_output_shapes

:@
╗
save/Assign_39Assignv/dense_2/kernel/Adam_1save/RestoreV2:39*
use_locking(*
T0*#
_class
loc:@v/dense_2/kernel*
validate_shape(*
_output_shapes

:@
Х
save/restore_shardNoOp^save/Assign^save/Assign_1^save/Assign_10^save/Assign_11^save/Assign_12^save/Assign_13^save/Assign_14^save/Assign_15^save/Assign_16^save/Assign_17^save/Assign_18^save/Assign_19^save/Assign_2^save/Assign_20^save/Assign_21^save/Assign_22^save/Assign_23^save/Assign_24^save/Assign_25^save/Assign_26^save/Assign_27^save/Assign_28^save/Assign_29^save/Assign_3^save/Assign_30^save/Assign_31^save/Assign_32^save/Assign_33^save/Assign_34^save/Assign_35^save/Assign_36^save/Assign_37^save/Assign_38^save/Assign_39^save/Assign_4^save/Assign_5^save/Assign_6^save/Assign_7^save/Assign_8^save/Assign_9
-
save/restore_allNoOp^save/restore_shard
[
save_1/filename/inputConst*
valueB Bmodel*
dtype0*
_output_shapes
: 
r
save_1/filenamePlaceholderWithDefaultsave_1/filename/input*
shape: *
dtype0*
_output_shapes
: 
i
save_1/ConstPlaceholderWithDefaultsave_1/filename*
_output_shapes
: *
shape: *
dtype0
є
save_1/StringJoin/inputs_1Const*<
value3B1 B+_temp_61a93a6431bc44509d0b5e0022b99b4e/part*
dtype0*
_output_shapes
: 
{
save_1/StringJoin
StringJoinsave_1/Constsave_1/StringJoin/inputs_1*
N*
_output_shapes
: *
	separator 
S
save_1/num_shardsConst*
dtype0*
_output_shapes
: *
value	B :
^
save_1/ShardedFilename/shardConst*
value	B : *
dtype0*
_output_shapes
: 
Ё
save_1/ShardedFilenameShardedFilenamesave_1/StringJoinsave_1/ShardedFilename/shardsave_1/num_shards*
_output_shapes
: 
і
save_1/SaveV2/tensor_namesConst*╗
value▒B«(Bbeta1_powerBbeta1_power_1Bbeta2_powerBbeta2_power_1Bpi/dense/biasBpi/dense/bias/AdamBpi/dense/bias/Adam_1Bpi/dense/kernelBpi/dense/kernel/AdamBpi/dense/kernel/Adam_1Bpi/dense_1/biasBpi/dense_1/bias/AdamBpi/dense_1/bias/Adam_1Bpi/dense_1/kernelBpi/dense_1/kernel/AdamBpi/dense_1/kernel/Adam_1Bpi/dense_2/biasBpi/dense_2/bias/AdamBpi/dense_2/bias/Adam_1Bpi/dense_2/kernelBpi/dense_2/kernel/AdamBpi/dense_2/kernel/Adam_1Bv/dense/biasBv/dense/bias/AdamBv/dense/bias/Adam_1Bv/dense/kernelBv/dense/kernel/AdamBv/dense/kernel/Adam_1Bv/dense_1/biasBv/dense_1/bias/AdamBv/dense_1/bias/Adam_1Bv/dense_1/kernelBv/dense_1/kernel/AdamBv/dense_1/kernel/Adam_1Bv/dense_2/biasBv/dense_2/bias/AdamBv/dense_2/bias/Adam_1Bv/dense_2/kernelBv/dense_2/kernel/AdamBv/dense_2/kernel/Adam_1*
dtype0*
_output_shapes
:(
х
save_1/SaveV2/shape_and_slicesConst*c
valueZBX(B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B *
dtype0*
_output_shapes
:(
╔
save_1/SaveV2SaveV2save_1/ShardedFilenamesave_1/SaveV2/tensor_namessave_1/SaveV2/shape_and_slicesbeta1_powerbeta1_power_1beta2_powerbeta2_power_1pi/dense/biaspi/dense/bias/Adampi/dense/bias/Adam_1pi/dense/kernelpi/dense/kernel/Adampi/dense/kernel/Adam_1pi/dense_1/biaspi/dense_1/bias/Adampi/dense_1/bias/Adam_1pi/dense_1/kernelpi/dense_1/kernel/Adampi/dense_1/kernel/Adam_1pi/dense_2/biaspi/dense_2/bias/Adampi/dense_2/bias/Adam_1pi/dense_2/kernelpi/dense_2/kernel/Adampi/dense_2/kernel/Adam_1v/dense/biasv/dense/bias/Adamv/dense/bias/Adam_1v/dense/kernelv/dense/kernel/Adamv/dense/kernel/Adam_1v/dense_1/biasv/dense_1/bias/Adamv/dense_1/bias/Adam_1v/dense_1/kernelv/dense_1/kernel/Adamv/dense_1/kernel/Adam_1v/dense_2/biasv/dense_2/bias/Adamv/dense_2/bias/Adam_1v/dense_2/kernelv/dense_2/kernel/Adamv/dense_2/kernel/Adam_1*6
dtypes,
*2(
Ў
save_1/control_dependencyIdentitysave_1/ShardedFilename^save_1/SaveV2*
T0*)
_class
loc:@save_1/ShardedFilename*
_output_shapes
: 
Б
-save_1/MergeV2Checkpoints/checkpoint_prefixesPacksave_1/ShardedFilename^save_1/control_dependency*
T0*

axis *
N*
_output_shapes
:
Ѓ
save_1/MergeV2CheckpointsMergeV2Checkpoints-save_1/MergeV2Checkpoints/checkpoint_prefixessave_1/Const*
delete_old_dirs(
ѓ
save_1/IdentityIdentitysave_1/Const^save_1/MergeV2Checkpoints^save_1/control_dependency*
T0*
_output_shapes
: 
Ї
save_1/RestoreV2/tensor_namesConst*╗
value▒B«(Bbeta1_powerBbeta1_power_1Bbeta2_powerBbeta2_power_1Bpi/dense/biasBpi/dense/bias/AdamBpi/dense/bias/Adam_1Bpi/dense/kernelBpi/dense/kernel/AdamBpi/dense/kernel/Adam_1Bpi/dense_1/biasBpi/dense_1/bias/AdamBpi/dense_1/bias/Adam_1Bpi/dense_1/kernelBpi/dense_1/kernel/AdamBpi/dense_1/kernel/Adam_1Bpi/dense_2/biasBpi/dense_2/bias/AdamBpi/dense_2/bias/Adam_1Bpi/dense_2/kernelBpi/dense_2/kernel/AdamBpi/dense_2/kernel/Adam_1Bv/dense/biasBv/dense/bias/AdamBv/dense/bias/Adam_1Bv/dense/kernelBv/dense/kernel/AdamBv/dense/kernel/Adam_1Bv/dense_1/biasBv/dense_1/bias/AdamBv/dense_1/bias/Adam_1Bv/dense_1/kernelBv/dense_1/kernel/AdamBv/dense_1/kernel/Adam_1Bv/dense_2/biasBv/dense_2/bias/AdamBv/dense_2/bias/Adam_1Bv/dense_2/kernelBv/dense_2/kernel/AdamBv/dense_2/kernel/Adam_1*
dtype0*
_output_shapes
:(
И
!save_1/RestoreV2/shape_and_slicesConst*c
valueZBX(B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B *
dtype0*
_output_shapes
:(
я
save_1/RestoreV2	RestoreV2save_1/Constsave_1/RestoreV2/tensor_names!save_1/RestoreV2/shape_and_slices*Х
_output_shapesБ
а::::::::::::::::::::::::::::::::::::::::*6
dtypes,
*2(
б
save_1/AssignAssignbeta1_powersave_1/RestoreV2*
T0* 
_class
loc:@pi/dense/bias*
validate_shape(*
_output_shapes
: *
use_locking(
Д
save_1/Assign_1Assignbeta1_power_1save_1/RestoreV2:1*
T0*
_class
loc:@v/dense/bias*
validate_shape(*
_output_shapes
: *
use_locking(
д
save_1/Assign_2Assignbeta2_powersave_1/RestoreV2:2*
use_locking(*
T0* 
_class
loc:@pi/dense/bias*
validate_shape(*
_output_shapes
: 
Д
save_1/Assign_3Assignbeta2_power_1save_1/RestoreV2:3*
use_locking(*
T0*
_class
loc:@v/dense/bias*
validate_shape(*
_output_shapes
: 
г
save_1/Assign_4Assignpi/dense/biassave_1/RestoreV2:4*
use_locking(*
T0* 
_class
loc:@pi/dense/bias*
validate_shape(*
_output_shapes
:@
▒
save_1/Assign_5Assignpi/dense/bias/Adamsave_1/RestoreV2:5*
T0* 
_class
loc:@pi/dense/bias*
validate_shape(*
_output_shapes
:@*
use_locking(
│
save_1/Assign_6Assignpi/dense/bias/Adam_1save_1/RestoreV2:6*
use_locking(*
T0* 
_class
loc:@pi/dense/bias*
validate_shape(*
_output_shapes
:@
х
save_1/Assign_7Assignpi/dense/kernelsave_1/RestoreV2:7*
use_locking(*
T0*"
_class
loc:@pi/dense/kernel*
validate_shape(*
_output_shapes
:	Є@
║
save_1/Assign_8Assignpi/dense/kernel/Adamsave_1/RestoreV2:8*
use_locking(*
T0*"
_class
loc:@pi/dense/kernel*
validate_shape(*
_output_shapes
:	Є@
╝
save_1/Assign_9Assignpi/dense/kernel/Adam_1save_1/RestoreV2:9*
use_locking(*
T0*"
_class
loc:@pi/dense/kernel*
validate_shape(*
_output_shapes
:	Є@
▓
save_1/Assign_10Assignpi/dense_1/biassave_1/RestoreV2:10*
T0*"
_class
loc:@pi/dense_1/bias*
validate_shape(*
_output_shapes
:@*
use_locking(
и
save_1/Assign_11Assignpi/dense_1/bias/Adamsave_1/RestoreV2:11*
use_locking(*
T0*"
_class
loc:@pi/dense_1/bias*
validate_shape(*
_output_shapes
:@
╣
save_1/Assign_12Assignpi/dense_1/bias/Adam_1save_1/RestoreV2:12*
validate_shape(*
_output_shapes
:@*
use_locking(*
T0*"
_class
loc:@pi/dense_1/bias
║
save_1/Assign_13Assignpi/dense_1/kernelsave_1/RestoreV2:13*
use_locking(*
T0*$
_class
loc:@pi/dense_1/kernel*
validate_shape(*
_output_shapes

:@@
┐
save_1/Assign_14Assignpi/dense_1/kernel/Adamsave_1/RestoreV2:14*
use_locking(*
T0*$
_class
loc:@pi/dense_1/kernel*
validate_shape(*
_output_shapes

:@@
┴
save_1/Assign_15Assignpi/dense_1/kernel/Adam_1save_1/RestoreV2:15*
validate_shape(*
_output_shapes

:@@*
use_locking(*
T0*$
_class
loc:@pi/dense_1/kernel
▓
save_1/Assign_16Assignpi/dense_2/biassave_1/RestoreV2:16*
use_locking(*
T0*"
_class
loc:@pi/dense_2/bias*
validate_shape(*
_output_shapes
:
и
save_1/Assign_17Assignpi/dense_2/bias/Adamsave_1/RestoreV2:17*
_output_shapes
:*
use_locking(*
T0*"
_class
loc:@pi/dense_2/bias*
validate_shape(
╣
save_1/Assign_18Assignpi/dense_2/bias/Adam_1save_1/RestoreV2:18*
use_locking(*
T0*"
_class
loc:@pi/dense_2/bias*
validate_shape(*
_output_shapes
:
║
save_1/Assign_19Assignpi/dense_2/kernelsave_1/RestoreV2:19*
T0*$
_class
loc:@pi/dense_2/kernel*
validate_shape(*
_output_shapes

:@*
use_locking(
┐
save_1/Assign_20Assignpi/dense_2/kernel/Adamsave_1/RestoreV2:20*
T0*$
_class
loc:@pi/dense_2/kernel*
validate_shape(*
_output_shapes

:@*
use_locking(
┴
save_1/Assign_21Assignpi/dense_2/kernel/Adam_1save_1/RestoreV2:21*
_output_shapes

:@*
use_locking(*
T0*$
_class
loc:@pi/dense_2/kernel*
validate_shape(
г
save_1/Assign_22Assignv/dense/biassave_1/RestoreV2:22*
use_locking(*
T0*
_class
loc:@v/dense/bias*
validate_shape(*
_output_shapes
:@
▒
save_1/Assign_23Assignv/dense/bias/Adamsave_1/RestoreV2:23*
use_locking(*
T0*
_class
loc:@v/dense/bias*
validate_shape(*
_output_shapes
:@
│
save_1/Assign_24Assignv/dense/bias/Adam_1save_1/RestoreV2:24*
validate_shape(*
_output_shapes
:@*
use_locking(*
T0*
_class
loc:@v/dense/bias
х
save_1/Assign_25Assignv/dense/kernelsave_1/RestoreV2:25*
use_locking(*
T0*!
_class
loc:@v/dense/kernel*
validate_shape(*
_output_shapes
:	Є@
║
save_1/Assign_26Assignv/dense/kernel/Adamsave_1/RestoreV2:26*
use_locking(*
T0*!
_class
loc:@v/dense/kernel*
validate_shape(*
_output_shapes
:	Є@
╝
save_1/Assign_27Assignv/dense/kernel/Adam_1save_1/RestoreV2:27*
T0*!
_class
loc:@v/dense/kernel*
validate_shape(*
_output_shapes
:	Є@*
use_locking(
░
save_1/Assign_28Assignv/dense_1/biassave_1/RestoreV2:28*
use_locking(*
T0*!
_class
loc:@v/dense_1/bias*
validate_shape(*
_output_shapes
:@
х
save_1/Assign_29Assignv/dense_1/bias/Adamsave_1/RestoreV2:29*
use_locking(*
T0*!
_class
loc:@v/dense_1/bias*
validate_shape(*
_output_shapes
:@
и
save_1/Assign_30Assignv/dense_1/bias/Adam_1save_1/RestoreV2:30*!
_class
loc:@v/dense_1/bias*
validate_shape(*
_output_shapes
:@*
use_locking(*
T0
И
save_1/Assign_31Assignv/dense_1/kernelsave_1/RestoreV2:31*
use_locking(*
T0*#
_class
loc:@v/dense_1/kernel*
validate_shape(*
_output_shapes

:@@
й
save_1/Assign_32Assignv/dense_1/kernel/Adamsave_1/RestoreV2:32*
T0*#
_class
loc:@v/dense_1/kernel*
validate_shape(*
_output_shapes

:@@*
use_locking(
┐
save_1/Assign_33Assignv/dense_1/kernel/Adam_1save_1/RestoreV2:33*
use_locking(*
T0*#
_class
loc:@v/dense_1/kernel*
validate_shape(*
_output_shapes

:@@
░
save_1/Assign_34Assignv/dense_2/biassave_1/RestoreV2:34*
use_locking(*
T0*!
_class
loc:@v/dense_2/bias*
validate_shape(*
_output_shapes
:
х
save_1/Assign_35Assignv/dense_2/bias/Adamsave_1/RestoreV2:35*
use_locking(*
T0*!
_class
loc:@v/dense_2/bias*
validate_shape(*
_output_shapes
:
и
save_1/Assign_36Assignv/dense_2/bias/Adam_1save_1/RestoreV2:36*
use_locking(*
T0*!
_class
loc:@v/dense_2/bias*
validate_shape(*
_output_shapes
:
И
save_1/Assign_37Assignv/dense_2/kernelsave_1/RestoreV2:37*
_output_shapes

:@*
use_locking(*
T0*#
_class
loc:@v/dense_2/kernel*
validate_shape(
й
save_1/Assign_38Assignv/dense_2/kernel/Adamsave_1/RestoreV2:38*#
_class
loc:@v/dense_2/kernel*
validate_shape(*
_output_shapes

:@*
use_locking(*
T0
┐
save_1/Assign_39Assignv/dense_2/kernel/Adam_1save_1/RestoreV2:39*
use_locking(*
T0*#
_class
loc:@v/dense_2/kernel*
validate_shape(*
_output_shapes

:@
ѕ
save_1/restore_shardNoOp^save_1/Assign^save_1/Assign_1^save_1/Assign_10^save_1/Assign_11^save_1/Assign_12^save_1/Assign_13^save_1/Assign_14^save_1/Assign_15^save_1/Assign_16^save_1/Assign_17^save_1/Assign_18^save_1/Assign_19^save_1/Assign_2^save_1/Assign_20^save_1/Assign_21^save_1/Assign_22^save_1/Assign_23^save_1/Assign_24^save_1/Assign_25^save_1/Assign_26^save_1/Assign_27^save_1/Assign_28^save_1/Assign_29^save_1/Assign_3^save_1/Assign_30^save_1/Assign_31^save_1/Assign_32^save_1/Assign_33^save_1/Assign_34^save_1/Assign_35^save_1/Assign_36^save_1/Assign_37^save_1/Assign_38^save_1/Assign_39^save_1/Assign_4^save_1/Assign_5^save_1/Assign_6^save_1/Assign_7^save_1/Assign_8^save_1/Assign_9
1
save_1/restore_allNoOp^save_1/restore_shard
[
save_2/filename/inputConst*
valueB Bmodel*
dtype0*
_output_shapes
: 
r
save_2/filenamePlaceholderWithDefaultsave_2/filename/input*
dtype0*
_output_shapes
: *
shape: 
i
save_2/ConstPlaceholderWithDefaultsave_2/filename*
dtype0*
_output_shapes
: *
shape: 
є
save_2/StringJoin/inputs_1Const*<
value3B1 B+_temp_3ae6d38df37c441baa8867e07c654507/part*
dtype0*
_output_shapes
: 
{
save_2/StringJoin
StringJoinsave_2/Constsave_2/StringJoin/inputs_1*
N*
_output_shapes
: *
	separator 
S
save_2/num_shardsConst*
_output_shapes
: *
value	B :*
dtype0
^
save_2/ShardedFilename/shardConst*
value	B : *
dtype0*
_output_shapes
: 
Ё
save_2/ShardedFilenameShardedFilenamesave_2/StringJoinsave_2/ShardedFilename/shardsave_2/num_shards*
_output_shapes
: 
і
save_2/SaveV2/tensor_namesConst*╗
value▒B«(Bbeta1_powerBbeta1_power_1Bbeta2_powerBbeta2_power_1Bpi/dense/biasBpi/dense/bias/AdamBpi/dense/bias/Adam_1Bpi/dense/kernelBpi/dense/kernel/AdamBpi/dense/kernel/Adam_1Bpi/dense_1/biasBpi/dense_1/bias/AdamBpi/dense_1/bias/Adam_1Bpi/dense_1/kernelBpi/dense_1/kernel/AdamBpi/dense_1/kernel/Adam_1Bpi/dense_2/biasBpi/dense_2/bias/AdamBpi/dense_2/bias/Adam_1Bpi/dense_2/kernelBpi/dense_2/kernel/AdamBpi/dense_2/kernel/Adam_1Bv/dense/biasBv/dense/bias/AdamBv/dense/bias/Adam_1Bv/dense/kernelBv/dense/kernel/AdamBv/dense/kernel/Adam_1Bv/dense_1/biasBv/dense_1/bias/AdamBv/dense_1/bias/Adam_1Bv/dense_1/kernelBv/dense_1/kernel/AdamBv/dense_1/kernel/Adam_1Bv/dense_2/biasBv/dense_2/bias/AdamBv/dense_2/bias/Adam_1Bv/dense_2/kernelBv/dense_2/kernel/AdamBv/dense_2/kernel/Adam_1*
dtype0*
_output_shapes
:(
х
save_2/SaveV2/shape_and_slicesConst*c
valueZBX(B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B *
dtype0*
_output_shapes
:(
╔
save_2/SaveV2SaveV2save_2/ShardedFilenamesave_2/SaveV2/tensor_namessave_2/SaveV2/shape_and_slicesbeta1_powerbeta1_power_1beta2_powerbeta2_power_1pi/dense/biaspi/dense/bias/Adampi/dense/bias/Adam_1pi/dense/kernelpi/dense/kernel/Adampi/dense/kernel/Adam_1pi/dense_1/biaspi/dense_1/bias/Adampi/dense_1/bias/Adam_1pi/dense_1/kernelpi/dense_1/kernel/Adampi/dense_1/kernel/Adam_1pi/dense_2/biaspi/dense_2/bias/Adampi/dense_2/bias/Adam_1pi/dense_2/kernelpi/dense_2/kernel/Adampi/dense_2/kernel/Adam_1v/dense/biasv/dense/bias/Adamv/dense/bias/Adam_1v/dense/kernelv/dense/kernel/Adamv/dense/kernel/Adam_1v/dense_1/biasv/dense_1/bias/Adamv/dense_1/bias/Adam_1v/dense_1/kernelv/dense_1/kernel/Adamv/dense_1/kernel/Adam_1v/dense_2/biasv/dense_2/bias/Adamv/dense_2/bias/Adam_1v/dense_2/kernelv/dense_2/kernel/Adamv/dense_2/kernel/Adam_1*6
dtypes,
*2(
Ў
save_2/control_dependencyIdentitysave_2/ShardedFilename^save_2/SaveV2*
T0*)
_class
loc:@save_2/ShardedFilename*
_output_shapes
: 
Б
-save_2/MergeV2Checkpoints/checkpoint_prefixesPacksave_2/ShardedFilename^save_2/control_dependency*
T0*

axis *
N*
_output_shapes
:
Ѓ
save_2/MergeV2CheckpointsMergeV2Checkpoints-save_2/MergeV2Checkpoints/checkpoint_prefixessave_2/Const*
delete_old_dirs(
ѓ
save_2/IdentityIdentitysave_2/Const^save_2/MergeV2Checkpoints^save_2/control_dependency*
T0*
_output_shapes
: 
Ї
save_2/RestoreV2/tensor_namesConst*╗
value▒B«(Bbeta1_powerBbeta1_power_1Bbeta2_powerBbeta2_power_1Bpi/dense/biasBpi/dense/bias/AdamBpi/dense/bias/Adam_1Bpi/dense/kernelBpi/dense/kernel/AdamBpi/dense/kernel/Adam_1Bpi/dense_1/biasBpi/dense_1/bias/AdamBpi/dense_1/bias/Adam_1Bpi/dense_1/kernelBpi/dense_1/kernel/AdamBpi/dense_1/kernel/Adam_1Bpi/dense_2/biasBpi/dense_2/bias/AdamBpi/dense_2/bias/Adam_1Bpi/dense_2/kernelBpi/dense_2/kernel/AdamBpi/dense_2/kernel/Adam_1Bv/dense/biasBv/dense/bias/AdamBv/dense/bias/Adam_1Bv/dense/kernelBv/dense/kernel/AdamBv/dense/kernel/Adam_1Bv/dense_1/biasBv/dense_1/bias/AdamBv/dense_1/bias/Adam_1Bv/dense_1/kernelBv/dense_1/kernel/AdamBv/dense_1/kernel/Adam_1Bv/dense_2/biasBv/dense_2/bias/AdamBv/dense_2/bias/Adam_1Bv/dense_2/kernelBv/dense_2/kernel/AdamBv/dense_2/kernel/Adam_1*
dtype0*
_output_shapes
:(
И
!save_2/RestoreV2/shape_and_slicesConst*c
valueZBX(B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B *
dtype0*
_output_shapes
:(
я
save_2/RestoreV2	RestoreV2save_2/Constsave_2/RestoreV2/tensor_names!save_2/RestoreV2/shape_and_slices*6
dtypes,
*2(*Х
_output_shapesБ
а::::::::::::::::::::::::::::::::::::::::
б
save_2/AssignAssignbeta1_powersave_2/RestoreV2* 
_class
loc:@pi/dense/bias*
validate_shape(*
_output_shapes
: *
use_locking(*
T0
Д
save_2/Assign_1Assignbeta1_power_1save_2/RestoreV2:1*
T0*
_class
loc:@v/dense/bias*
validate_shape(*
_output_shapes
: *
use_locking(
д
save_2/Assign_2Assignbeta2_powersave_2/RestoreV2:2*
validate_shape(*
_output_shapes
: *
use_locking(*
T0* 
_class
loc:@pi/dense/bias
Д
save_2/Assign_3Assignbeta2_power_1save_2/RestoreV2:3*
validate_shape(*
_output_shapes
: *
use_locking(*
T0*
_class
loc:@v/dense/bias
г
save_2/Assign_4Assignpi/dense/biassave_2/RestoreV2:4*
use_locking(*
T0* 
_class
loc:@pi/dense/bias*
validate_shape(*
_output_shapes
:@
▒
save_2/Assign_5Assignpi/dense/bias/Adamsave_2/RestoreV2:5*
validate_shape(*
_output_shapes
:@*
use_locking(*
T0* 
_class
loc:@pi/dense/bias
│
save_2/Assign_6Assignpi/dense/bias/Adam_1save_2/RestoreV2:6*
use_locking(*
T0* 
_class
loc:@pi/dense/bias*
validate_shape(*
_output_shapes
:@
х
save_2/Assign_7Assignpi/dense/kernelsave_2/RestoreV2:7*
use_locking(*
T0*"
_class
loc:@pi/dense/kernel*
validate_shape(*
_output_shapes
:	Є@
║
save_2/Assign_8Assignpi/dense/kernel/Adamsave_2/RestoreV2:8*
T0*"
_class
loc:@pi/dense/kernel*
validate_shape(*
_output_shapes
:	Є@*
use_locking(
╝
save_2/Assign_9Assignpi/dense/kernel/Adam_1save_2/RestoreV2:9*
_output_shapes
:	Є@*
use_locking(*
T0*"
_class
loc:@pi/dense/kernel*
validate_shape(
▓
save_2/Assign_10Assignpi/dense_1/biassave_2/RestoreV2:10*"
_class
loc:@pi/dense_1/bias*
validate_shape(*
_output_shapes
:@*
use_locking(*
T0
и
save_2/Assign_11Assignpi/dense_1/bias/Adamsave_2/RestoreV2:11*
use_locking(*
T0*"
_class
loc:@pi/dense_1/bias*
validate_shape(*
_output_shapes
:@
╣
save_2/Assign_12Assignpi/dense_1/bias/Adam_1save_2/RestoreV2:12*
T0*"
_class
loc:@pi/dense_1/bias*
validate_shape(*
_output_shapes
:@*
use_locking(
║
save_2/Assign_13Assignpi/dense_1/kernelsave_2/RestoreV2:13*
use_locking(*
T0*$
_class
loc:@pi/dense_1/kernel*
validate_shape(*
_output_shapes

:@@
┐
save_2/Assign_14Assignpi/dense_1/kernel/Adamsave_2/RestoreV2:14*$
_class
loc:@pi/dense_1/kernel*
validate_shape(*
_output_shapes

:@@*
use_locking(*
T0
┴
save_2/Assign_15Assignpi/dense_1/kernel/Adam_1save_2/RestoreV2:15*
use_locking(*
T0*$
_class
loc:@pi/dense_1/kernel*
validate_shape(*
_output_shapes

:@@
▓
save_2/Assign_16Assignpi/dense_2/biassave_2/RestoreV2:16*"
_class
loc:@pi/dense_2/bias*
validate_shape(*
_output_shapes
:*
use_locking(*
T0
и
save_2/Assign_17Assignpi/dense_2/bias/Adamsave_2/RestoreV2:17*
use_locking(*
T0*"
_class
loc:@pi/dense_2/bias*
validate_shape(*
_output_shapes
:
╣
save_2/Assign_18Assignpi/dense_2/bias/Adam_1save_2/RestoreV2:18*
use_locking(*
T0*"
_class
loc:@pi/dense_2/bias*
validate_shape(*
_output_shapes
:
║
save_2/Assign_19Assignpi/dense_2/kernelsave_2/RestoreV2:19*
_output_shapes

:@*
use_locking(*
T0*$
_class
loc:@pi/dense_2/kernel*
validate_shape(
┐
save_2/Assign_20Assignpi/dense_2/kernel/Adamsave_2/RestoreV2:20*
use_locking(*
T0*$
_class
loc:@pi/dense_2/kernel*
validate_shape(*
_output_shapes

:@
┴
save_2/Assign_21Assignpi/dense_2/kernel/Adam_1save_2/RestoreV2:21*
validate_shape(*
_output_shapes

:@*
use_locking(*
T0*$
_class
loc:@pi/dense_2/kernel
г
save_2/Assign_22Assignv/dense/biassave_2/RestoreV2:22*
use_locking(*
T0*
_class
loc:@v/dense/bias*
validate_shape(*
_output_shapes
:@
▒
save_2/Assign_23Assignv/dense/bias/Adamsave_2/RestoreV2:23*
validate_shape(*
_output_shapes
:@*
use_locking(*
T0*
_class
loc:@v/dense/bias
│
save_2/Assign_24Assignv/dense/bias/Adam_1save_2/RestoreV2:24*
_class
loc:@v/dense/bias*
validate_shape(*
_output_shapes
:@*
use_locking(*
T0
х
save_2/Assign_25Assignv/dense/kernelsave_2/RestoreV2:25*
use_locking(*
T0*!
_class
loc:@v/dense/kernel*
validate_shape(*
_output_shapes
:	Є@
║
save_2/Assign_26Assignv/dense/kernel/Adamsave_2/RestoreV2:26*
use_locking(*
T0*!
_class
loc:@v/dense/kernel*
validate_shape(*
_output_shapes
:	Є@
╝
save_2/Assign_27Assignv/dense/kernel/Adam_1save_2/RestoreV2:27*
use_locking(*
T0*!
_class
loc:@v/dense/kernel*
validate_shape(*
_output_shapes
:	Є@
░
save_2/Assign_28Assignv/dense_1/biassave_2/RestoreV2:28*
T0*!
_class
loc:@v/dense_1/bias*
validate_shape(*
_output_shapes
:@*
use_locking(
х
save_2/Assign_29Assignv/dense_1/bias/Adamsave_2/RestoreV2:29*
use_locking(*
T0*!
_class
loc:@v/dense_1/bias*
validate_shape(*
_output_shapes
:@
и
save_2/Assign_30Assignv/dense_1/bias/Adam_1save_2/RestoreV2:30*
validate_shape(*
_output_shapes
:@*
use_locking(*
T0*!
_class
loc:@v/dense_1/bias
И
save_2/Assign_31Assignv/dense_1/kernelsave_2/RestoreV2:31*
use_locking(*
T0*#
_class
loc:@v/dense_1/kernel*
validate_shape(*
_output_shapes

:@@
й
save_2/Assign_32Assignv/dense_1/kernel/Adamsave_2/RestoreV2:32*
use_locking(*
T0*#
_class
loc:@v/dense_1/kernel*
validate_shape(*
_output_shapes

:@@
┐
save_2/Assign_33Assignv/dense_1/kernel/Adam_1save_2/RestoreV2:33*
_output_shapes

:@@*
use_locking(*
T0*#
_class
loc:@v/dense_1/kernel*
validate_shape(
░
save_2/Assign_34Assignv/dense_2/biassave_2/RestoreV2:34*
T0*!
_class
loc:@v/dense_2/bias*
validate_shape(*
_output_shapes
:*
use_locking(
х
save_2/Assign_35Assignv/dense_2/bias/Adamsave_2/RestoreV2:35*
use_locking(*
T0*!
_class
loc:@v/dense_2/bias*
validate_shape(*
_output_shapes
:
и
save_2/Assign_36Assignv/dense_2/bias/Adam_1save_2/RestoreV2:36*
use_locking(*
T0*!
_class
loc:@v/dense_2/bias*
validate_shape(*
_output_shapes
:
И
save_2/Assign_37Assignv/dense_2/kernelsave_2/RestoreV2:37*
use_locking(*
T0*#
_class
loc:@v/dense_2/kernel*
validate_shape(*
_output_shapes

:@
й
save_2/Assign_38Assignv/dense_2/kernel/Adamsave_2/RestoreV2:38*
use_locking(*
T0*#
_class
loc:@v/dense_2/kernel*
validate_shape(*
_output_shapes

:@
┐
save_2/Assign_39Assignv/dense_2/kernel/Adam_1save_2/RestoreV2:39*#
_class
loc:@v/dense_2/kernel*
validate_shape(*
_output_shapes

:@*
use_locking(*
T0
ѕ
save_2/restore_shardNoOp^save_2/Assign^save_2/Assign_1^save_2/Assign_10^save_2/Assign_11^save_2/Assign_12^save_2/Assign_13^save_2/Assign_14^save_2/Assign_15^save_2/Assign_16^save_2/Assign_17^save_2/Assign_18^save_2/Assign_19^save_2/Assign_2^save_2/Assign_20^save_2/Assign_21^save_2/Assign_22^save_2/Assign_23^save_2/Assign_24^save_2/Assign_25^save_2/Assign_26^save_2/Assign_27^save_2/Assign_28^save_2/Assign_29^save_2/Assign_3^save_2/Assign_30^save_2/Assign_31^save_2/Assign_32^save_2/Assign_33^save_2/Assign_34^save_2/Assign_35^save_2/Assign_36^save_2/Assign_37^save_2/Assign_38^save_2/Assign_39^save_2/Assign_4^save_2/Assign_5^save_2/Assign_6^save_2/Assign_7^save_2/Assign_8^save_2/Assign_9
1
save_2/restore_allNoOp^save_2/restore_shard
[
save_3/filename/inputConst*
valueB Bmodel*
dtype0*
_output_shapes
: 
r
save_3/filenamePlaceholderWithDefaultsave_3/filename/input*
dtype0*
_output_shapes
: *
shape: 
i
save_3/ConstPlaceholderWithDefaultsave_3/filename*
shape: *
dtype0*
_output_shapes
: 
є
save_3/StringJoin/inputs_1Const*<
value3B1 B+_temp_85a3b2a07d824706bbcfac3623ab7f71/part*
dtype0*
_output_shapes
: 
{
save_3/StringJoin
StringJoinsave_3/Constsave_3/StringJoin/inputs_1*
_output_shapes
: *
	separator *
N
S
save_3/num_shardsConst*
value	B :*
dtype0*
_output_shapes
: 
^
save_3/ShardedFilename/shardConst*
value	B : *
dtype0*
_output_shapes
: 
Ё
save_3/ShardedFilenameShardedFilenamesave_3/StringJoinsave_3/ShardedFilename/shardsave_3/num_shards*
_output_shapes
: 
і
save_3/SaveV2/tensor_namesConst*
_output_shapes
:(*╗
value▒B«(Bbeta1_powerBbeta1_power_1Bbeta2_powerBbeta2_power_1Bpi/dense/biasBpi/dense/bias/AdamBpi/dense/bias/Adam_1Bpi/dense/kernelBpi/dense/kernel/AdamBpi/dense/kernel/Adam_1Bpi/dense_1/biasBpi/dense_1/bias/AdamBpi/dense_1/bias/Adam_1Bpi/dense_1/kernelBpi/dense_1/kernel/AdamBpi/dense_1/kernel/Adam_1Bpi/dense_2/biasBpi/dense_2/bias/AdamBpi/dense_2/bias/Adam_1Bpi/dense_2/kernelBpi/dense_2/kernel/AdamBpi/dense_2/kernel/Adam_1Bv/dense/biasBv/dense/bias/AdamBv/dense/bias/Adam_1Bv/dense/kernelBv/dense/kernel/AdamBv/dense/kernel/Adam_1Bv/dense_1/biasBv/dense_1/bias/AdamBv/dense_1/bias/Adam_1Bv/dense_1/kernelBv/dense_1/kernel/AdamBv/dense_1/kernel/Adam_1Bv/dense_2/biasBv/dense_2/bias/AdamBv/dense_2/bias/Adam_1Bv/dense_2/kernelBv/dense_2/kernel/AdamBv/dense_2/kernel/Adam_1*
dtype0
х
save_3/SaveV2/shape_and_slicesConst*c
valueZBX(B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B *
dtype0*
_output_shapes
:(
╔
save_3/SaveV2SaveV2save_3/ShardedFilenamesave_3/SaveV2/tensor_namessave_3/SaveV2/shape_and_slicesbeta1_powerbeta1_power_1beta2_powerbeta2_power_1pi/dense/biaspi/dense/bias/Adampi/dense/bias/Adam_1pi/dense/kernelpi/dense/kernel/Adampi/dense/kernel/Adam_1pi/dense_1/biaspi/dense_1/bias/Adampi/dense_1/bias/Adam_1pi/dense_1/kernelpi/dense_1/kernel/Adampi/dense_1/kernel/Adam_1pi/dense_2/biaspi/dense_2/bias/Adampi/dense_2/bias/Adam_1pi/dense_2/kernelpi/dense_2/kernel/Adampi/dense_2/kernel/Adam_1v/dense/biasv/dense/bias/Adamv/dense/bias/Adam_1v/dense/kernelv/dense/kernel/Adamv/dense/kernel/Adam_1v/dense_1/biasv/dense_1/bias/Adamv/dense_1/bias/Adam_1v/dense_1/kernelv/dense_1/kernel/Adamv/dense_1/kernel/Adam_1v/dense_2/biasv/dense_2/bias/Adamv/dense_2/bias/Adam_1v/dense_2/kernelv/dense_2/kernel/Adamv/dense_2/kernel/Adam_1*6
dtypes,
*2(
Ў
save_3/control_dependencyIdentitysave_3/ShardedFilename^save_3/SaveV2*
T0*)
_class
loc:@save_3/ShardedFilename*
_output_shapes
: 
Б
-save_3/MergeV2Checkpoints/checkpoint_prefixesPacksave_3/ShardedFilename^save_3/control_dependency*
T0*

axis *
N*
_output_shapes
:
Ѓ
save_3/MergeV2CheckpointsMergeV2Checkpoints-save_3/MergeV2Checkpoints/checkpoint_prefixessave_3/Const*
delete_old_dirs(
ѓ
save_3/IdentityIdentitysave_3/Const^save_3/MergeV2Checkpoints^save_3/control_dependency*
T0*
_output_shapes
: 
Ї
save_3/RestoreV2/tensor_namesConst*╗
value▒B«(Bbeta1_powerBbeta1_power_1Bbeta2_powerBbeta2_power_1Bpi/dense/biasBpi/dense/bias/AdamBpi/dense/bias/Adam_1Bpi/dense/kernelBpi/dense/kernel/AdamBpi/dense/kernel/Adam_1Bpi/dense_1/biasBpi/dense_1/bias/AdamBpi/dense_1/bias/Adam_1Bpi/dense_1/kernelBpi/dense_1/kernel/AdamBpi/dense_1/kernel/Adam_1Bpi/dense_2/biasBpi/dense_2/bias/AdamBpi/dense_2/bias/Adam_1Bpi/dense_2/kernelBpi/dense_2/kernel/AdamBpi/dense_2/kernel/Adam_1Bv/dense/biasBv/dense/bias/AdamBv/dense/bias/Adam_1Bv/dense/kernelBv/dense/kernel/AdamBv/dense/kernel/Adam_1Bv/dense_1/biasBv/dense_1/bias/AdamBv/dense_1/bias/Adam_1Bv/dense_1/kernelBv/dense_1/kernel/AdamBv/dense_1/kernel/Adam_1Bv/dense_2/biasBv/dense_2/bias/AdamBv/dense_2/bias/Adam_1Bv/dense_2/kernelBv/dense_2/kernel/AdamBv/dense_2/kernel/Adam_1*
dtype0*
_output_shapes
:(
И
!save_3/RestoreV2/shape_and_slicesConst*
_output_shapes
:(*c
valueZBX(B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B *
dtype0
я
save_3/RestoreV2	RestoreV2save_3/Constsave_3/RestoreV2/tensor_names!save_3/RestoreV2/shape_and_slices*6
dtypes,
*2(*Х
_output_shapesБ
а::::::::::::::::::::::::::::::::::::::::
б
save_3/AssignAssignbeta1_powersave_3/RestoreV2*
T0* 
_class
loc:@pi/dense/bias*
validate_shape(*
_output_shapes
: *
use_locking(
Д
save_3/Assign_1Assignbeta1_power_1save_3/RestoreV2:1*
use_locking(*
T0*
_class
loc:@v/dense/bias*
validate_shape(*
_output_shapes
: 
д
save_3/Assign_2Assignbeta2_powersave_3/RestoreV2:2*
use_locking(*
T0* 
_class
loc:@pi/dense/bias*
validate_shape(*
_output_shapes
: 
Д
save_3/Assign_3Assignbeta2_power_1save_3/RestoreV2:3*
use_locking(*
T0*
_class
loc:@v/dense/bias*
validate_shape(*
_output_shapes
: 
г
save_3/Assign_4Assignpi/dense/biassave_3/RestoreV2:4*
validate_shape(*
_output_shapes
:@*
use_locking(*
T0* 
_class
loc:@pi/dense/bias
▒
save_3/Assign_5Assignpi/dense/bias/Adamsave_3/RestoreV2:5*
use_locking(*
T0* 
_class
loc:@pi/dense/bias*
validate_shape(*
_output_shapes
:@
│
save_3/Assign_6Assignpi/dense/bias/Adam_1save_3/RestoreV2:6* 
_class
loc:@pi/dense/bias*
validate_shape(*
_output_shapes
:@*
use_locking(*
T0
х
save_3/Assign_7Assignpi/dense/kernelsave_3/RestoreV2:7*
use_locking(*
T0*"
_class
loc:@pi/dense/kernel*
validate_shape(*
_output_shapes
:	Є@
║
save_3/Assign_8Assignpi/dense/kernel/Adamsave_3/RestoreV2:8*
T0*"
_class
loc:@pi/dense/kernel*
validate_shape(*
_output_shapes
:	Є@*
use_locking(
╝
save_3/Assign_9Assignpi/dense/kernel/Adam_1save_3/RestoreV2:9*
use_locking(*
T0*"
_class
loc:@pi/dense/kernel*
validate_shape(*
_output_shapes
:	Є@
▓
save_3/Assign_10Assignpi/dense_1/biassave_3/RestoreV2:10*
use_locking(*
T0*"
_class
loc:@pi/dense_1/bias*
validate_shape(*
_output_shapes
:@
и
save_3/Assign_11Assignpi/dense_1/bias/Adamsave_3/RestoreV2:11*
use_locking(*
T0*"
_class
loc:@pi/dense_1/bias*
validate_shape(*
_output_shapes
:@
╣
save_3/Assign_12Assignpi/dense_1/bias/Adam_1save_3/RestoreV2:12*
use_locking(*
T0*"
_class
loc:@pi/dense_1/bias*
validate_shape(*
_output_shapes
:@
║
save_3/Assign_13Assignpi/dense_1/kernelsave_3/RestoreV2:13*
use_locking(*
T0*$
_class
loc:@pi/dense_1/kernel*
validate_shape(*
_output_shapes

:@@
┐
save_3/Assign_14Assignpi/dense_1/kernel/Adamsave_3/RestoreV2:14*
validate_shape(*
_output_shapes

:@@*
use_locking(*
T0*$
_class
loc:@pi/dense_1/kernel
┴
save_3/Assign_15Assignpi/dense_1/kernel/Adam_1save_3/RestoreV2:15*$
_class
loc:@pi/dense_1/kernel*
validate_shape(*
_output_shapes

:@@*
use_locking(*
T0
▓
save_3/Assign_16Assignpi/dense_2/biassave_3/RestoreV2:16*
validate_shape(*
_output_shapes
:*
use_locking(*
T0*"
_class
loc:@pi/dense_2/bias
и
save_3/Assign_17Assignpi/dense_2/bias/Adamsave_3/RestoreV2:17*
T0*"
_class
loc:@pi/dense_2/bias*
validate_shape(*
_output_shapes
:*
use_locking(
╣
save_3/Assign_18Assignpi/dense_2/bias/Adam_1save_3/RestoreV2:18*
T0*"
_class
loc:@pi/dense_2/bias*
validate_shape(*
_output_shapes
:*
use_locking(
║
save_3/Assign_19Assignpi/dense_2/kernelsave_3/RestoreV2:19*
use_locking(*
T0*$
_class
loc:@pi/dense_2/kernel*
validate_shape(*
_output_shapes

:@
┐
save_3/Assign_20Assignpi/dense_2/kernel/Adamsave_3/RestoreV2:20*
use_locking(*
T0*$
_class
loc:@pi/dense_2/kernel*
validate_shape(*
_output_shapes

:@
┴
save_3/Assign_21Assignpi/dense_2/kernel/Adam_1save_3/RestoreV2:21*
T0*$
_class
loc:@pi/dense_2/kernel*
validate_shape(*
_output_shapes

:@*
use_locking(
г
save_3/Assign_22Assignv/dense/biassave_3/RestoreV2:22*
_class
loc:@v/dense/bias*
validate_shape(*
_output_shapes
:@*
use_locking(*
T0
▒
save_3/Assign_23Assignv/dense/bias/Adamsave_3/RestoreV2:23*
use_locking(*
T0*
_class
loc:@v/dense/bias*
validate_shape(*
_output_shapes
:@
│
save_3/Assign_24Assignv/dense/bias/Adam_1save_3/RestoreV2:24*
use_locking(*
T0*
_class
loc:@v/dense/bias*
validate_shape(*
_output_shapes
:@
х
save_3/Assign_25Assignv/dense/kernelsave_3/RestoreV2:25*
use_locking(*
T0*!
_class
loc:@v/dense/kernel*
validate_shape(*
_output_shapes
:	Є@
║
save_3/Assign_26Assignv/dense/kernel/Adamsave_3/RestoreV2:26*
use_locking(*
T0*!
_class
loc:@v/dense/kernel*
validate_shape(*
_output_shapes
:	Є@
╝
save_3/Assign_27Assignv/dense/kernel/Adam_1save_3/RestoreV2:27*
use_locking(*
T0*!
_class
loc:@v/dense/kernel*
validate_shape(*
_output_shapes
:	Є@
░
save_3/Assign_28Assignv/dense_1/biassave_3/RestoreV2:28*
use_locking(*
T0*!
_class
loc:@v/dense_1/bias*
validate_shape(*
_output_shapes
:@
х
save_3/Assign_29Assignv/dense_1/bias/Adamsave_3/RestoreV2:29*
use_locking(*
T0*!
_class
loc:@v/dense_1/bias*
validate_shape(*
_output_shapes
:@
и
save_3/Assign_30Assignv/dense_1/bias/Adam_1save_3/RestoreV2:30*
use_locking(*
T0*!
_class
loc:@v/dense_1/bias*
validate_shape(*
_output_shapes
:@
И
save_3/Assign_31Assignv/dense_1/kernelsave_3/RestoreV2:31*
validate_shape(*
_output_shapes

:@@*
use_locking(*
T0*#
_class
loc:@v/dense_1/kernel
й
save_3/Assign_32Assignv/dense_1/kernel/Adamsave_3/RestoreV2:32*
use_locking(*
T0*#
_class
loc:@v/dense_1/kernel*
validate_shape(*
_output_shapes

:@@
┐
save_3/Assign_33Assignv/dense_1/kernel/Adam_1save_3/RestoreV2:33*
use_locking(*
T0*#
_class
loc:@v/dense_1/kernel*
validate_shape(*
_output_shapes

:@@
░
save_3/Assign_34Assignv/dense_2/biassave_3/RestoreV2:34*
use_locking(*
T0*!
_class
loc:@v/dense_2/bias*
validate_shape(*
_output_shapes
:
х
save_3/Assign_35Assignv/dense_2/bias/Adamsave_3/RestoreV2:35*
validate_shape(*
_output_shapes
:*
use_locking(*
T0*!
_class
loc:@v/dense_2/bias
и
save_3/Assign_36Assignv/dense_2/bias/Adam_1save_3/RestoreV2:36*!
_class
loc:@v/dense_2/bias*
validate_shape(*
_output_shapes
:*
use_locking(*
T0
И
save_3/Assign_37Assignv/dense_2/kernelsave_3/RestoreV2:37*
use_locking(*
T0*#
_class
loc:@v/dense_2/kernel*
validate_shape(*
_output_shapes

:@
й
save_3/Assign_38Assignv/dense_2/kernel/Adamsave_3/RestoreV2:38*
T0*#
_class
loc:@v/dense_2/kernel*
validate_shape(*
_output_shapes

:@*
use_locking(
┐
save_3/Assign_39Assignv/dense_2/kernel/Adam_1save_3/RestoreV2:39*
T0*#
_class
loc:@v/dense_2/kernel*
validate_shape(*
_output_shapes

:@*
use_locking(
ѕ
save_3/restore_shardNoOp^save_3/Assign^save_3/Assign_1^save_3/Assign_10^save_3/Assign_11^save_3/Assign_12^save_3/Assign_13^save_3/Assign_14^save_3/Assign_15^save_3/Assign_16^save_3/Assign_17^save_3/Assign_18^save_3/Assign_19^save_3/Assign_2^save_3/Assign_20^save_3/Assign_21^save_3/Assign_22^save_3/Assign_23^save_3/Assign_24^save_3/Assign_25^save_3/Assign_26^save_3/Assign_27^save_3/Assign_28^save_3/Assign_29^save_3/Assign_3^save_3/Assign_30^save_3/Assign_31^save_3/Assign_32^save_3/Assign_33^save_3/Assign_34^save_3/Assign_35^save_3/Assign_36^save_3/Assign_37^save_3/Assign_38^save_3/Assign_39^save_3/Assign_4^save_3/Assign_5^save_3/Assign_6^save_3/Assign_7^save_3/Assign_8^save_3/Assign_9
1
save_3/restore_allNoOp^save_3/restore_shard
[
save_4/filename/inputConst*
dtype0*
_output_shapes
: *
valueB Bmodel
r
save_4/filenamePlaceholderWithDefaultsave_4/filename/input*
dtype0*
_output_shapes
: *
shape: 
i
save_4/ConstPlaceholderWithDefaultsave_4/filename*
dtype0*
_output_shapes
: *
shape: 
є
save_4/StringJoin/inputs_1Const*<
value3B1 B+_temp_bce154ed8bc54526aaef1f4328b8e1ea/part*
dtype0*
_output_shapes
: 
{
save_4/StringJoin
StringJoinsave_4/Constsave_4/StringJoin/inputs_1*
	separator *
N*
_output_shapes
: 
S
save_4/num_shardsConst*
_output_shapes
: *
value	B :*
dtype0
^
save_4/ShardedFilename/shardConst*
_output_shapes
: *
value	B : *
dtype0
Ё
save_4/ShardedFilenameShardedFilenamesave_4/StringJoinsave_4/ShardedFilename/shardsave_4/num_shards*
_output_shapes
: 
і
save_4/SaveV2/tensor_namesConst*╗
value▒B«(Bbeta1_powerBbeta1_power_1Bbeta2_powerBbeta2_power_1Bpi/dense/biasBpi/dense/bias/AdamBpi/dense/bias/Adam_1Bpi/dense/kernelBpi/dense/kernel/AdamBpi/dense/kernel/Adam_1Bpi/dense_1/biasBpi/dense_1/bias/AdamBpi/dense_1/bias/Adam_1Bpi/dense_1/kernelBpi/dense_1/kernel/AdamBpi/dense_1/kernel/Adam_1Bpi/dense_2/biasBpi/dense_2/bias/AdamBpi/dense_2/bias/Adam_1Bpi/dense_2/kernelBpi/dense_2/kernel/AdamBpi/dense_2/kernel/Adam_1Bv/dense/biasBv/dense/bias/AdamBv/dense/bias/Adam_1Bv/dense/kernelBv/dense/kernel/AdamBv/dense/kernel/Adam_1Bv/dense_1/biasBv/dense_1/bias/AdamBv/dense_1/bias/Adam_1Bv/dense_1/kernelBv/dense_1/kernel/AdamBv/dense_1/kernel/Adam_1Bv/dense_2/biasBv/dense_2/bias/AdamBv/dense_2/bias/Adam_1Bv/dense_2/kernelBv/dense_2/kernel/AdamBv/dense_2/kernel/Adam_1*
dtype0*
_output_shapes
:(
х
save_4/SaveV2/shape_and_slicesConst*
_output_shapes
:(*c
valueZBX(B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B *
dtype0
╔
save_4/SaveV2SaveV2save_4/ShardedFilenamesave_4/SaveV2/tensor_namessave_4/SaveV2/shape_and_slicesbeta1_powerbeta1_power_1beta2_powerbeta2_power_1pi/dense/biaspi/dense/bias/Adampi/dense/bias/Adam_1pi/dense/kernelpi/dense/kernel/Adampi/dense/kernel/Adam_1pi/dense_1/biaspi/dense_1/bias/Adampi/dense_1/bias/Adam_1pi/dense_1/kernelpi/dense_1/kernel/Adampi/dense_1/kernel/Adam_1pi/dense_2/biaspi/dense_2/bias/Adampi/dense_2/bias/Adam_1pi/dense_2/kernelpi/dense_2/kernel/Adampi/dense_2/kernel/Adam_1v/dense/biasv/dense/bias/Adamv/dense/bias/Adam_1v/dense/kernelv/dense/kernel/Adamv/dense/kernel/Adam_1v/dense_1/biasv/dense_1/bias/Adamv/dense_1/bias/Adam_1v/dense_1/kernelv/dense_1/kernel/Adamv/dense_1/kernel/Adam_1v/dense_2/biasv/dense_2/bias/Adamv/dense_2/bias/Adam_1v/dense_2/kernelv/dense_2/kernel/Adamv/dense_2/kernel/Adam_1*6
dtypes,
*2(
Ў
save_4/control_dependencyIdentitysave_4/ShardedFilename^save_4/SaveV2*
T0*)
_class
loc:@save_4/ShardedFilename*
_output_shapes
: 
Б
-save_4/MergeV2Checkpoints/checkpoint_prefixesPacksave_4/ShardedFilename^save_4/control_dependency*
T0*

axis *
N*
_output_shapes
:
Ѓ
save_4/MergeV2CheckpointsMergeV2Checkpoints-save_4/MergeV2Checkpoints/checkpoint_prefixessave_4/Const*
delete_old_dirs(
ѓ
save_4/IdentityIdentitysave_4/Const^save_4/MergeV2Checkpoints^save_4/control_dependency*
T0*
_output_shapes
: 
Ї
save_4/RestoreV2/tensor_namesConst*
_output_shapes
:(*╗
value▒B«(Bbeta1_powerBbeta1_power_1Bbeta2_powerBbeta2_power_1Bpi/dense/biasBpi/dense/bias/AdamBpi/dense/bias/Adam_1Bpi/dense/kernelBpi/dense/kernel/AdamBpi/dense/kernel/Adam_1Bpi/dense_1/biasBpi/dense_1/bias/AdamBpi/dense_1/bias/Adam_1Bpi/dense_1/kernelBpi/dense_1/kernel/AdamBpi/dense_1/kernel/Adam_1Bpi/dense_2/biasBpi/dense_2/bias/AdamBpi/dense_2/bias/Adam_1Bpi/dense_2/kernelBpi/dense_2/kernel/AdamBpi/dense_2/kernel/Adam_1Bv/dense/biasBv/dense/bias/AdamBv/dense/bias/Adam_1Bv/dense/kernelBv/dense/kernel/AdamBv/dense/kernel/Adam_1Bv/dense_1/biasBv/dense_1/bias/AdamBv/dense_1/bias/Adam_1Bv/dense_1/kernelBv/dense_1/kernel/AdamBv/dense_1/kernel/Adam_1Bv/dense_2/biasBv/dense_2/bias/AdamBv/dense_2/bias/Adam_1Bv/dense_2/kernelBv/dense_2/kernel/AdamBv/dense_2/kernel/Adam_1*
dtype0
И
!save_4/RestoreV2/shape_and_slicesConst*c
valueZBX(B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B *
dtype0*
_output_shapes
:(
я
save_4/RestoreV2	RestoreV2save_4/Constsave_4/RestoreV2/tensor_names!save_4/RestoreV2/shape_and_slices*Х
_output_shapesБ
а::::::::::::::::::::::::::::::::::::::::*6
dtypes,
*2(
б
save_4/AssignAssignbeta1_powersave_4/RestoreV2*
use_locking(*
T0* 
_class
loc:@pi/dense/bias*
validate_shape(*
_output_shapes
: 
Д
save_4/Assign_1Assignbeta1_power_1save_4/RestoreV2:1*
validate_shape(*
_output_shapes
: *
use_locking(*
T0*
_class
loc:@v/dense/bias
д
save_4/Assign_2Assignbeta2_powersave_4/RestoreV2:2*
use_locking(*
T0* 
_class
loc:@pi/dense/bias*
validate_shape(*
_output_shapes
: 
Д
save_4/Assign_3Assignbeta2_power_1save_4/RestoreV2:3*
use_locking(*
T0*
_class
loc:@v/dense/bias*
validate_shape(*
_output_shapes
: 
г
save_4/Assign_4Assignpi/dense/biassave_4/RestoreV2:4*
_output_shapes
:@*
use_locking(*
T0* 
_class
loc:@pi/dense/bias*
validate_shape(
▒
save_4/Assign_5Assignpi/dense/bias/Adamsave_4/RestoreV2:5*
use_locking(*
T0* 
_class
loc:@pi/dense/bias*
validate_shape(*
_output_shapes
:@
│
save_4/Assign_6Assignpi/dense/bias/Adam_1save_4/RestoreV2:6*
T0* 
_class
loc:@pi/dense/bias*
validate_shape(*
_output_shapes
:@*
use_locking(
х
save_4/Assign_7Assignpi/dense/kernelsave_4/RestoreV2:7*
use_locking(*
T0*"
_class
loc:@pi/dense/kernel*
validate_shape(*
_output_shapes
:	Є@
║
save_4/Assign_8Assignpi/dense/kernel/Adamsave_4/RestoreV2:8*
use_locking(*
T0*"
_class
loc:@pi/dense/kernel*
validate_shape(*
_output_shapes
:	Є@
╝
save_4/Assign_9Assignpi/dense/kernel/Adam_1save_4/RestoreV2:9*
use_locking(*
T0*"
_class
loc:@pi/dense/kernel*
validate_shape(*
_output_shapes
:	Є@
▓
save_4/Assign_10Assignpi/dense_1/biassave_4/RestoreV2:10*"
_class
loc:@pi/dense_1/bias*
validate_shape(*
_output_shapes
:@*
use_locking(*
T0
и
save_4/Assign_11Assignpi/dense_1/bias/Adamsave_4/RestoreV2:11*
T0*"
_class
loc:@pi/dense_1/bias*
validate_shape(*
_output_shapes
:@*
use_locking(
╣
save_4/Assign_12Assignpi/dense_1/bias/Adam_1save_4/RestoreV2:12*
validate_shape(*
_output_shapes
:@*
use_locking(*
T0*"
_class
loc:@pi/dense_1/bias
║
save_4/Assign_13Assignpi/dense_1/kernelsave_4/RestoreV2:13*
use_locking(*
T0*$
_class
loc:@pi/dense_1/kernel*
validate_shape(*
_output_shapes

:@@
┐
save_4/Assign_14Assignpi/dense_1/kernel/Adamsave_4/RestoreV2:14*
_output_shapes

:@@*
use_locking(*
T0*$
_class
loc:@pi/dense_1/kernel*
validate_shape(
┴
save_4/Assign_15Assignpi/dense_1/kernel/Adam_1save_4/RestoreV2:15*
use_locking(*
T0*$
_class
loc:@pi/dense_1/kernel*
validate_shape(*
_output_shapes

:@@
▓
save_4/Assign_16Assignpi/dense_2/biassave_4/RestoreV2:16*
use_locking(*
T0*"
_class
loc:@pi/dense_2/bias*
validate_shape(*
_output_shapes
:
и
save_4/Assign_17Assignpi/dense_2/bias/Adamsave_4/RestoreV2:17*
use_locking(*
T0*"
_class
loc:@pi/dense_2/bias*
validate_shape(*
_output_shapes
:
╣
save_4/Assign_18Assignpi/dense_2/bias/Adam_1save_4/RestoreV2:18*
use_locking(*
T0*"
_class
loc:@pi/dense_2/bias*
validate_shape(*
_output_shapes
:
║
save_4/Assign_19Assignpi/dense_2/kernelsave_4/RestoreV2:19*
use_locking(*
T0*$
_class
loc:@pi/dense_2/kernel*
validate_shape(*
_output_shapes

:@
┐
save_4/Assign_20Assignpi/dense_2/kernel/Adamsave_4/RestoreV2:20*
validate_shape(*
_output_shapes

:@*
use_locking(*
T0*$
_class
loc:@pi/dense_2/kernel
┴
save_4/Assign_21Assignpi/dense_2/kernel/Adam_1save_4/RestoreV2:21*
use_locking(*
T0*$
_class
loc:@pi/dense_2/kernel*
validate_shape(*
_output_shapes

:@
г
save_4/Assign_22Assignv/dense/biassave_4/RestoreV2:22*
validate_shape(*
_output_shapes
:@*
use_locking(*
T0*
_class
loc:@v/dense/bias
▒
save_4/Assign_23Assignv/dense/bias/Adamsave_4/RestoreV2:23*
use_locking(*
T0*
_class
loc:@v/dense/bias*
validate_shape(*
_output_shapes
:@
│
save_4/Assign_24Assignv/dense/bias/Adam_1save_4/RestoreV2:24*
T0*
_class
loc:@v/dense/bias*
validate_shape(*
_output_shapes
:@*
use_locking(
х
save_4/Assign_25Assignv/dense/kernelsave_4/RestoreV2:25*
_output_shapes
:	Є@*
use_locking(*
T0*!
_class
loc:@v/dense/kernel*
validate_shape(
║
save_4/Assign_26Assignv/dense/kernel/Adamsave_4/RestoreV2:26*
use_locking(*
T0*!
_class
loc:@v/dense/kernel*
validate_shape(*
_output_shapes
:	Є@
╝
save_4/Assign_27Assignv/dense/kernel/Adam_1save_4/RestoreV2:27*
use_locking(*
T0*!
_class
loc:@v/dense/kernel*
validate_shape(*
_output_shapes
:	Є@
░
save_4/Assign_28Assignv/dense_1/biassave_4/RestoreV2:28*
use_locking(*
T0*!
_class
loc:@v/dense_1/bias*
validate_shape(*
_output_shapes
:@
х
save_4/Assign_29Assignv/dense_1/bias/Adamsave_4/RestoreV2:29*
T0*!
_class
loc:@v/dense_1/bias*
validate_shape(*
_output_shapes
:@*
use_locking(
и
save_4/Assign_30Assignv/dense_1/bias/Adam_1save_4/RestoreV2:30*
validate_shape(*
_output_shapes
:@*
use_locking(*
T0*!
_class
loc:@v/dense_1/bias
И
save_4/Assign_31Assignv/dense_1/kernelsave_4/RestoreV2:31*
use_locking(*
T0*#
_class
loc:@v/dense_1/kernel*
validate_shape(*
_output_shapes

:@@
й
save_4/Assign_32Assignv/dense_1/kernel/Adamsave_4/RestoreV2:32*
_output_shapes

:@@*
use_locking(*
T0*#
_class
loc:@v/dense_1/kernel*
validate_shape(
┐
save_4/Assign_33Assignv/dense_1/kernel/Adam_1save_4/RestoreV2:33*
_output_shapes

:@@*
use_locking(*
T0*#
_class
loc:@v/dense_1/kernel*
validate_shape(
░
save_4/Assign_34Assignv/dense_2/biassave_4/RestoreV2:34*
use_locking(*
T0*!
_class
loc:@v/dense_2/bias*
validate_shape(*
_output_shapes
:
х
save_4/Assign_35Assignv/dense_2/bias/Adamsave_4/RestoreV2:35*
use_locking(*
T0*!
_class
loc:@v/dense_2/bias*
validate_shape(*
_output_shapes
:
и
save_4/Assign_36Assignv/dense_2/bias/Adam_1save_4/RestoreV2:36*
_output_shapes
:*
use_locking(*
T0*!
_class
loc:@v/dense_2/bias*
validate_shape(
И
save_4/Assign_37Assignv/dense_2/kernelsave_4/RestoreV2:37*
use_locking(*
T0*#
_class
loc:@v/dense_2/kernel*
validate_shape(*
_output_shapes

:@
й
save_4/Assign_38Assignv/dense_2/kernel/Adamsave_4/RestoreV2:38*
_output_shapes

:@*
use_locking(*
T0*#
_class
loc:@v/dense_2/kernel*
validate_shape(
┐
save_4/Assign_39Assignv/dense_2/kernel/Adam_1save_4/RestoreV2:39*
T0*#
_class
loc:@v/dense_2/kernel*
validate_shape(*
_output_shapes

:@*
use_locking(
ѕ
save_4/restore_shardNoOp^save_4/Assign^save_4/Assign_1^save_4/Assign_10^save_4/Assign_11^save_4/Assign_12^save_4/Assign_13^save_4/Assign_14^save_4/Assign_15^save_4/Assign_16^save_4/Assign_17^save_4/Assign_18^save_4/Assign_19^save_4/Assign_2^save_4/Assign_20^save_4/Assign_21^save_4/Assign_22^save_4/Assign_23^save_4/Assign_24^save_4/Assign_25^save_4/Assign_26^save_4/Assign_27^save_4/Assign_28^save_4/Assign_29^save_4/Assign_3^save_4/Assign_30^save_4/Assign_31^save_4/Assign_32^save_4/Assign_33^save_4/Assign_34^save_4/Assign_35^save_4/Assign_36^save_4/Assign_37^save_4/Assign_38^save_4/Assign_39^save_4/Assign_4^save_4/Assign_5^save_4/Assign_6^save_4/Assign_7^save_4/Assign_8^save_4/Assign_9
1
save_4/restore_allNoOp^save_4/restore_shard
[
save_5/filename/inputConst*
valueB Bmodel*
dtype0*
_output_shapes
: 
r
save_5/filenamePlaceholderWithDefaultsave_5/filename/input*
shape: *
dtype0*
_output_shapes
: 
i
save_5/ConstPlaceholderWithDefaultsave_5/filename*
dtype0*
_output_shapes
: *
shape: 
є
save_5/StringJoin/inputs_1Const*<
value3B1 B+_temp_eb65bf79396d4055a9546e5db74f5492/part*
dtype0*
_output_shapes
: 
{
save_5/StringJoin
StringJoinsave_5/Constsave_5/StringJoin/inputs_1*
N*
_output_shapes
: *
	separator 
S
save_5/num_shardsConst*
_output_shapes
: *
value	B :*
dtype0
^
save_5/ShardedFilename/shardConst*
value	B : *
dtype0*
_output_shapes
: 
Ё
save_5/ShardedFilenameShardedFilenamesave_5/StringJoinsave_5/ShardedFilename/shardsave_5/num_shards*
_output_shapes
: 
і
save_5/SaveV2/tensor_namesConst*
dtype0*
_output_shapes
:(*╗
value▒B«(Bbeta1_powerBbeta1_power_1Bbeta2_powerBbeta2_power_1Bpi/dense/biasBpi/dense/bias/AdamBpi/dense/bias/Adam_1Bpi/dense/kernelBpi/dense/kernel/AdamBpi/dense/kernel/Adam_1Bpi/dense_1/biasBpi/dense_1/bias/AdamBpi/dense_1/bias/Adam_1Bpi/dense_1/kernelBpi/dense_1/kernel/AdamBpi/dense_1/kernel/Adam_1Bpi/dense_2/biasBpi/dense_2/bias/AdamBpi/dense_2/bias/Adam_1Bpi/dense_2/kernelBpi/dense_2/kernel/AdamBpi/dense_2/kernel/Adam_1Bv/dense/biasBv/dense/bias/AdamBv/dense/bias/Adam_1Bv/dense/kernelBv/dense/kernel/AdamBv/dense/kernel/Adam_1Bv/dense_1/biasBv/dense_1/bias/AdamBv/dense_1/bias/Adam_1Bv/dense_1/kernelBv/dense_1/kernel/AdamBv/dense_1/kernel/Adam_1Bv/dense_2/biasBv/dense_2/bias/AdamBv/dense_2/bias/Adam_1Bv/dense_2/kernelBv/dense_2/kernel/AdamBv/dense_2/kernel/Adam_1
х
save_5/SaveV2/shape_and_slicesConst*c
valueZBX(B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B *
dtype0*
_output_shapes
:(
╔
save_5/SaveV2SaveV2save_5/ShardedFilenamesave_5/SaveV2/tensor_namessave_5/SaveV2/shape_and_slicesbeta1_powerbeta1_power_1beta2_powerbeta2_power_1pi/dense/biaspi/dense/bias/Adampi/dense/bias/Adam_1pi/dense/kernelpi/dense/kernel/Adampi/dense/kernel/Adam_1pi/dense_1/biaspi/dense_1/bias/Adampi/dense_1/bias/Adam_1pi/dense_1/kernelpi/dense_1/kernel/Adampi/dense_1/kernel/Adam_1pi/dense_2/biaspi/dense_2/bias/Adampi/dense_2/bias/Adam_1pi/dense_2/kernelpi/dense_2/kernel/Adampi/dense_2/kernel/Adam_1v/dense/biasv/dense/bias/Adamv/dense/bias/Adam_1v/dense/kernelv/dense/kernel/Adamv/dense/kernel/Adam_1v/dense_1/biasv/dense_1/bias/Adamv/dense_1/bias/Adam_1v/dense_1/kernelv/dense_1/kernel/Adamv/dense_1/kernel/Adam_1v/dense_2/biasv/dense_2/bias/Adamv/dense_2/bias/Adam_1v/dense_2/kernelv/dense_2/kernel/Adamv/dense_2/kernel/Adam_1*6
dtypes,
*2(
Ў
save_5/control_dependencyIdentitysave_5/ShardedFilename^save_5/SaveV2*)
_class
loc:@save_5/ShardedFilename*
_output_shapes
: *
T0
Б
-save_5/MergeV2Checkpoints/checkpoint_prefixesPacksave_5/ShardedFilename^save_5/control_dependency*
T0*

axis *
N*
_output_shapes
:
Ѓ
save_5/MergeV2CheckpointsMergeV2Checkpoints-save_5/MergeV2Checkpoints/checkpoint_prefixessave_5/Const*
delete_old_dirs(
ѓ
save_5/IdentityIdentitysave_5/Const^save_5/MergeV2Checkpoints^save_5/control_dependency*
_output_shapes
: *
T0
Ї
save_5/RestoreV2/tensor_namesConst*
_output_shapes
:(*╗
value▒B«(Bbeta1_powerBbeta1_power_1Bbeta2_powerBbeta2_power_1Bpi/dense/biasBpi/dense/bias/AdamBpi/dense/bias/Adam_1Bpi/dense/kernelBpi/dense/kernel/AdamBpi/dense/kernel/Adam_1Bpi/dense_1/biasBpi/dense_1/bias/AdamBpi/dense_1/bias/Adam_1Bpi/dense_1/kernelBpi/dense_1/kernel/AdamBpi/dense_1/kernel/Adam_1Bpi/dense_2/biasBpi/dense_2/bias/AdamBpi/dense_2/bias/Adam_1Bpi/dense_2/kernelBpi/dense_2/kernel/AdamBpi/dense_2/kernel/Adam_1Bv/dense/biasBv/dense/bias/AdamBv/dense/bias/Adam_1Bv/dense/kernelBv/dense/kernel/AdamBv/dense/kernel/Adam_1Bv/dense_1/biasBv/dense_1/bias/AdamBv/dense_1/bias/Adam_1Bv/dense_1/kernelBv/dense_1/kernel/AdamBv/dense_1/kernel/Adam_1Bv/dense_2/biasBv/dense_2/bias/AdamBv/dense_2/bias/Adam_1Bv/dense_2/kernelBv/dense_2/kernel/AdamBv/dense_2/kernel/Adam_1*
dtype0
И
!save_5/RestoreV2/shape_and_slicesConst*
_output_shapes
:(*c
valueZBX(B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B *
dtype0
я
save_5/RestoreV2	RestoreV2save_5/Constsave_5/RestoreV2/tensor_names!save_5/RestoreV2/shape_and_slices*Х
_output_shapesБ
а::::::::::::::::::::::::::::::::::::::::*6
dtypes,
*2(
б
save_5/AssignAssignbeta1_powersave_5/RestoreV2*
validate_shape(*
_output_shapes
: *
use_locking(*
T0* 
_class
loc:@pi/dense/bias
Д
save_5/Assign_1Assignbeta1_power_1save_5/RestoreV2:1*
use_locking(*
T0*
_class
loc:@v/dense/bias*
validate_shape(*
_output_shapes
: 
д
save_5/Assign_2Assignbeta2_powersave_5/RestoreV2:2*
use_locking(*
T0* 
_class
loc:@pi/dense/bias*
validate_shape(*
_output_shapes
: 
Д
save_5/Assign_3Assignbeta2_power_1save_5/RestoreV2:3*
T0*
_class
loc:@v/dense/bias*
validate_shape(*
_output_shapes
: *
use_locking(
г
save_5/Assign_4Assignpi/dense/biassave_5/RestoreV2:4*
T0* 
_class
loc:@pi/dense/bias*
validate_shape(*
_output_shapes
:@*
use_locking(
▒
save_5/Assign_5Assignpi/dense/bias/Adamsave_5/RestoreV2:5*
use_locking(*
T0* 
_class
loc:@pi/dense/bias*
validate_shape(*
_output_shapes
:@
│
save_5/Assign_6Assignpi/dense/bias/Adam_1save_5/RestoreV2:6*
_output_shapes
:@*
use_locking(*
T0* 
_class
loc:@pi/dense/bias*
validate_shape(
х
save_5/Assign_7Assignpi/dense/kernelsave_5/RestoreV2:7*"
_class
loc:@pi/dense/kernel*
validate_shape(*
_output_shapes
:	Є@*
use_locking(*
T0
║
save_5/Assign_8Assignpi/dense/kernel/Adamsave_5/RestoreV2:8*
use_locking(*
T0*"
_class
loc:@pi/dense/kernel*
validate_shape(*
_output_shapes
:	Є@
╝
save_5/Assign_9Assignpi/dense/kernel/Adam_1save_5/RestoreV2:9*"
_class
loc:@pi/dense/kernel*
validate_shape(*
_output_shapes
:	Є@*
use_locking(*
T0
▓
save_5/Assign_10Assignpi/dense_1/biassave_5/RestoreV2:10*
use_locking(*
T0*"
_class
loc:@pi/dense_1/bias*
validate_shape(*
_output_shapes
:@
и
save_5/Assign_11Assignpi/dense_1/bias/Adamsave_5/RestoreV2:11*
T0*"
_class
loc:@pi/dense_1/bias*
validate_shape(*
_output_shapes
:@*
use_locking(
╣
save_5/Assign_12Assignpi/dense_1/bias/Adam_1save_5/RestoreV2:12*
validate_shape(*
_output_shapes
:@*
use_locking(*
T0*"
_class
loc:@pi/dense_1/bias
║
save_5/Assign_13Assignpi/dense_1/kernelsave_5/RestoreV2:13*
use_locking(*
T0*$
_class
loc:@pi/dense_1/kernel*
validate_shape(*
_output_shapes

:@@
┐
save_5/Assign_14Assignpi/dense_1/kernel/Adamsave_5/RestoreV2:14*$
_class
loc:@pi/dense_1/kernel*
validate_shape(*
_output_shapes

:@@*
use_locking(*
T0
┴
save_5/Assign_15Assignpi/dense_1/kernel/Adam_1save_5/RestoreV2:15*
use_locking(*
T0*$
_class
loc:@pi/dense_1/kernel*
validate_shape(*
_output_shapes

:@@
▓
save_5/Assign_16Assignpi/dense_2/biassave_5/RestoreV2:16*
use_locking(*
T0*"
_class
loc:@pi/dense_2/bias*
validate_shape(*
_output_shapes
:
и
save_5/Assign_17Assignpi/dense_2/bias/Adamsave_5/RestoreV2:17*"
_class
loc:@pi/dense_2/bias*
validate_shape(*
_output_shapes
:*
use_locking(*
T0
╣
save_5/Assign_18Assignpi/dense_2/bias/Adam_1save_5/RestoreV2:18*"
_class
loc:@pi/dense_2/bias*
validate_shape(*
_output_shapes
:*
use_locking(*
T0
║
save_5/Assign_19Assignpi/dense_2/kernelsave_5/RestoreV2:19*$
_class
loc:@pi/dense_2/kernel*
validate_shape(*
_output_shapes

:@*
use_locking(*
T0
┐
save_5/Assign_20Assignpi/dense_2/kernel/Adamsave_5/RestoreV2:20*
_output_shapes

:@*
use_locking(*
T0*$
_class
loc:@pi/dense_2/kernel*
validate_shape(
┴
save_5/Assign_21Assignpi/dense_2/kernel/Adam_1save_5/RestoreV2:21*
validate_shape(*
_output_shapes

:@*
use_locking(*
T0*$
_class
loc:@pi/dense_2/kernel
г
save_5/Assign_22Assignv/dense/biassave_5/RestoreV2:22*
use_locking(*
T0*
_class
loc:@v/dense/bias*
validate_shape(*
_output_shapes
:@
▒
save_5/Assign_23Assignv/dense/bias/Adamsave_5/RestoreV2:23*
_class
loc:@v/dense/bias*
validate_shape(*
_output_shapes
:@*
use_locking(*
T0
│
save_5/Assign_24Assignv/dense/bias/Adam_1save_5/RestoreV2:24*
_output_shapes
:@*
use_locking(*
T0*
_class
loc:@v/dense/bias*
validate_shape(
х
save_5/Assign_25Assignv/dense/kernelsave_5/RestoreV2:25*
use_locking(*
T0*!
_class
loc:@v/dense/kernel*
validate_shape(*
_output_shapes
:	Є@
║
save_5/Assign_26Assignv/dense/kernel/Adamsave_5/RestoreV2:26*
use_locking(*
T0*!
_class
loc:@v/dense/kernel*
validate_shape(*
_output_shapes
:	Є@
╝
save_5/Assign_27Assignv/dense/kernel/Adam_1save_5/RestoreV2:27*
use_locking(*
T0*!
_class
loc:@v/dense/kernel*
validate_shape(*
_output_shapes
:	Є@
░
save_5/Assign_28Assignv/dense_1/biassave_5/RestoreV2:28*
use_locking(*
T0*!
_class
loc:@v/dense_1/bias*
validate_shape(*
_output_shapes
:@
х
save_5/Assign_29Assignv/dense_1/bias/Adamsave_5/RestoreV2:29*
use_locking(*
T0*!
_class
loc:@v/dense_1/bias*
validate_shape(*
_output_shapes
:@
и
save_5/Assign_30Assignv/dense_1/bias/Adam_1save_5/RestoreV2:30*
_output_shapes
:@*
use_locking(*
T0*!
_class
loc:@v/dense_1/bias*
validate_shape(
И
save_5/Assign_31Assignv/dense_1/kernelsave_5/RestoreV2:31*
validate_shape(*
_output_shapes

:@@*
use_locking(*
T0*#
_class
loc:@v/dense_1/kernel
й
save_5/Assign_32Assignv/dense_1/kernel/Adamsave_5/RestoreV2:32*
T0*#
_class
loc:@v/dense_1/kernel*
validate_shape(*
_output_shapes

:@@*
use_locking(
┐
save_5/Assign_33Assignv/dense_1/kernel/Adam_1save_5/RestoreV2:33*
use_locking(*
T0*#
_class
loc:@v/dense_1/kernel*
validate_shape(*
_output_shapes

:@@
░
save_5/Assign_34Assignv/dense_2/biassave_5/RestoreV2:34*
validate_shape(*
_output_shapes
:*
use_locking(*
T0*!
_class
loc:@v/dense_2/bias
х
save_5/Assign_35Assignv/dense_2/bias/Adamsave_5/RestoreV2:35*
_output_shapes
:*
use_locking(*
T0*!
_class
loc:@v/dense_2/bias*
validate_shape(
и
save_5/Assign_36Assignv/dense_2/bias/Adam_1save_5/RestoreV2:36*
use_locking(*
T0*!
_class
loc:@v/dense_2/bias*
validate_shape(*
_output_shapes
:
И
save_5/Assign_37Assignv/dense_2/kernelsave_5/RestoreV2:37*
use_locking(*
T0*#
_class
loc:@v/dense_2/kernel*
validate_shape(*
_output_shapes

:@
й
save_5/Assign_38Assignv/dense_2/kernel/Adamsave_5/RestoreV2:38*#
_class
loc:@v/dense_2/kernel*
validate_shape(*
_output_shapes

:@*
use_locking(*
T0
┐
save_5/Assign_39Assignv/dense_2/kernel/Adam_1save_5/RestoreV2:39*
use_locking(*
T0*#
_class
loc:@v/dense_2/kernel*
validate_shape(*
_output_shapes

:@
ѕ
save_5/restore_shardNoOp^save_5/Assign^save_5/Assign_1^save_5/Assign_10^save_5/Assign_11^save_5/Assign_12^save_5/Assign_13^save_5/Assign_14^save_5/Assign_15^save_5/Assign_16^save_5/Assign_17^save_5/Assign_18^save_5/Assign_19^save_5/Assign_2^save_5/Assign_20^save_5/Assign_21^save_5/Assign_22^save_5/Assign_23^save_5/Assign_24^save_5/Assign_25^save_5/Assign_26^save_5/Assign_27^save_5/Assign_28^save_5/Assign_29^save_5/Assign_3^save_5/Assign_30^save_5/Assign_31^save_5/Assign_32^save_5/Assign_33^save_5/Assign_34^save_5/Assign_35^save_5/Assign_36^save_5/Assign_37^save_5/Assign_38^save_5/Assign_39^save_5/Assign_4^save_5/Assign_5^save_5/Assign_6^save_5/Assign_7^save_5/Assign_8^save_5/Assign_9
1
save_5/restore_allNoOp^save_5/restore_shard
[
save_6/filename/inputConst*
_output_shapes
: *
valueB Bmodel*
dtype0
r
save_6/filenamePlaceholderWithDefaultsave_6/filename/input*
dtype0*
_output_shapes
: *
shape: 
i
save_6/ConstPlaceholderWithDefaultsave_6/filename*
dtype0*
_output_shapes
: *
shape: 
є
save_6/StringJoin/inputs_1Const*
dtype0*
_output_shapes
: *<
value3B1 B+_temp_d59d0cf86dc74751a8d77d2b3c11e5f5/part
{
save_6/StringJoin
StringJoinsave_6/Constsave_6/StringJoin/inputs_1*
	separator *
N*
_output_shapes
: 
S
save_6/num_shardsConst*
value	B :*
dtype0*
_output_shapes
: 
^
save_6/ShardedFilename/shardConst*
dtype0*
_output_shapes
: *
value	B : 
Ё
save_6/ShardedFilenameShardedFilenamesave_6/StringJoinsave_6/ShardedFilename/shardsave_6/num_shards*
_output_shapes
: 
і
save_6/SaveV2/tensor_namesConst*╗
value▒B«(Bbeta1_powerBbeta1_power_1Bbeta2_powerBbeta2_power_1Bpi/dense/biasBpi/dense/bias/AdamBpi/dense/bias/Adam_1Bpi/dense/kernelBpi/dense/kernel/AdamBpi/dense/kernel/Adam_1Bpi/dense_1/biasBpi/dense_1/bias/AdamBpi/dense_1/bias/Adam_1Bpi/dense_1/kernelBpi/dense_1/kernel/AdamBpi/dense_1/kernel/Adam_1Bpi/dense_2/biasBpi/dense_2/bias/AdamBpi/dense_2/bias/Adam_1Bpi/dense_2/kernelBpi/dense_2/kernel/AdamBpi/dense_2/kernel/Adam_1Bv/dense/biasBv/dense/bias/AdamBv/dense/bias/Adam_1Bv/dense/kernelBv/dense/kernel/AdamBv/dense/kernel/Adam_1Bv/dense_1/biasBv/dense_1/bias/AdamBv/dense_1/bias/Adam_1Bv/dense_1/kernelBv/dense_1/kernel/AdamBv/dense_1/kernel/Adam_1Bv/dense_2/biasBv/dense_2/bias/AdamBv/dense_2/bias/Adam_1Bv/dense_2/kernelBv/dense_2/kernel/AdamBv/dense_2/kernel/Adam_1*
dtype0*
_output_shapes
:(
х
save_6/SaveV2/shape_and_slicesConst*
_output_shapes
:(*c
valueZBX(B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B *
dtype0
╔
save_6/SaveV2SaveV2save_6/ShardedFilenamesave_6/SaveV2/tensor_namessave_6/SaveV2/shape_and_slicesbeta1_powerbeta1_power_1beta2_powerbeta2_power_1pi/dense/biaspi/dense/bias/Adampi/dense/bias/Adam_1pi/dense/kernelpi/dense/kernel/Adampi/dense/kernel/Adam_1pi/dense_1/biaspi/dense_1/bias/Adampi/dense_1/bias/Adam_1pi/dense_1/kernelpi/dense_1/kernel/Adampi/dense_1/kernel/Adam_1pi/dense_2/biaspi/dense_2/bias/Adampi/dense_2/bias/Adam_1pi/dense_2/kernelpi/dense_2/kernel/Adampi/dense_2/kernel/Adam_1v/dense/biasv/dense/bias/Adamv/dense/bias/Adam_1v/dense/kernelv/dense/kernel/Adamv/dense/kernel/Adam_1v/dense_1/biasv/dense_1/bias/Adamv/dense_1/bias/Adam_1v/dense_1/kernelv/dense_1/kernel/Adamv/dense_1/kernel/Adam_1v/dense_2/biasv/dense_2/bias/Adamv/dense_2/bias/Adam_1v/dense_2/kernelv/dense_2/kernel/Adamv/dense_2/kernel/Adam_1*6
dtypes,
*2(
Ў
save_6/control_dependencyIdentitysave_6/ShardedFilename^save_6/SaveV2*)
_class
loc:@save_6/ShardedFilename*
_output_shapes
: *
T0
Б
-save_6/MergeV2Checkpoints/checkpoint_prefixesPacksave_6/ShardedFilename^save_6/control_dependency*
N*
_output_shapes
:*
T0*

axis 
Ѓ
save_6/MergeV2CheckpointsMergeV2Checkpoints-save_6/MergeV2Checkpoints/checkpoint_prefixessave_6/Const*
delete_old_dirs(
ѓ
save_6/IdentityIdentitysave_6/Const^save_6/MergeV2Checkpoints^save_6/control_dependency*
T0*
_output_shapes
: 
Ї
save_6/RestoreV2/tensor_namesConst*╗
value▒B«(Bbeta1_powerBbeta1_power_1Bbeta2_powerBbeta2_power_1Bpi/dense/biasBpi/dense/bias/AdamBpi/dense/bias/Adam_1Bpi/dense/kernelBpi/dense/kernel/AdamBpi/dense/kernel/Adam_1Bpi/dense_1/biasBpi/dense_1/bias/AdamBpi/dense_1/bias/Adam_1Bpi/dense_1/kernelBpi/dense_1/kernel/AdamBpi/dense_1/kernel/Adam_1Bpi/dense_2/biasBpi/dense_2/bias/AdamBpi/dense_2/bias/Adam_1Bpi/dense_2/kernelBpi/dense_2/kernel/AdamBpi/dense_2/kernel/Adam_1Bv/dense/biasBv/dense/bias/AdamBv/dense/bias/Adam_1Bv/dense/kernelBv/dense/kernel/AdamBv/dense/kernel/Adam_1Bv/dense_1/biasBv/dense_1/bias/AdamBv/dense_1/bias/Adam_1Bv/dense_1/kernelBv/dense_1/kernel/AdamBv/dense_1/kernel/Adam_1Bv/dense_2/biasBv/dense_2/bias/AdamBv/dense_2/bias/Adam_1Bv/dense_2/kernelBv/dense_2/kernel/AdamBv/dense_2/kernel/Adam_1*
dtype0*
_output_shapes
:(
И
!save_6/RestoreV2/shape_and_slicesConst*c
valueZBX(B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B *
dtype0*
_output_shapes
:(
я
save_6/RestoreV2	RestoreV2save_6/Constsave_6/RestoreV2/tensor_names!save_6/RestoreV2/shape_and_slices*6
dtypes,
*2(*Х
_output_shapesБ
а::::::::::::::::::::::::::::::::::::::::
б
save_6/AssignAssignbeta1_powersave_6/RestoreV2*
use_locking(*
T0* 
_class
loc:@pi/dense/bias*
validate_shape(*
_output_shapes
: 
Д
save_6/Assign_1Assignbeta1_power_1save_6/RestoreV2:1*
use_locking(*
T0*
_class
loc:@v/dense/bias*
validate_shape(*
_output_shapes
: 
д
save_6/Assign_2Assignbeta2_powersave_6/RestoreV2:2*
_output_shapes
: *
use_locking(*
T0* 
_class
loc:@pi/dense/bias*
validate_shape(
Д
save_6/Assign_3Assignbeta2_power_1save_6/RestoreV2:3*
_class
loc:@v/dense/bias*
validate_shape(*
_output_shapes
: *
use_locking(*
T0
г
save_6/Assign_4Assignpi/dense/biassave_6/RestoreV2:4*
use_locking(*
T0* 
_class
loc:@pi/dense/bias*
validate_shape(*
_output_shapes
:@
▒
save_6/Assign_5Assignpi/dense/bias/Adamsave_6/RestoreV2:5*
use_locking(*
T0* 
_class
loc:@pi/dense/bias*
validate_shape(*
_output_shapes
:@
│
save_6/Assign_6Assignpi/dense/bias/Adam_1save_6/RestoreV2:6* 
_class
loc:@pi/dense/bias*
validate_shape(*
_output_shapes
:@*
use_locking(*
T0
х
save_6/Assign_7Assignpi/dense/kernelsave_6/RestoreV2:7*
_output_shapes
:	Є@*
use_locking(*
T0*"
_class
loc:@pi/dense/kernel*
validate_shape(
║
save_6/Assign_8Assignpi/dense/kernel/Adamsave_6/RestoreV2:8*
T0*"
_class
loc:@pi/dense/kernel*
validate_shape(*
_output_shapes
:	Є@*
use_locking(
╝
save_6/Assign_9Assignpi/dense/kernel/Adam_1save_6/RestoreV2:9*
use_locking(*
T0*"
_class
loc:@pi/dense/kernel*
validate_shape(*
_output_shapes
:	Є@
▓
save_6/Assign_10Assignpi/dense_1/biassave_6/RestoreV2:10*
validate_shape(*
_output_shapes
:@*
use_locking(*
T0*"
_class
loc:@pi/dense_1/bias
и
save_6/Assign_11Assignpi/dense_1/bias/Adamsave_6/RestoreV2:11*
T0*"
_class
loc:@pi/dense_1/bias*
validate_shape(*
_output_shapes
:@*
use_locking(
╣
save_6/Assign_12Assignpi/dense_1/bias/Adam_1save_6/RestoreV2:12*
use_locking(*
T0*"
_class
loc:@pi/dense_1/bias*
validate_shape(*
_output_shapes
:@
║
save_6/Assign_13Assignpi/dense_1/kernelsave_6/RestoreV2:13*
use_locking(*
T0*$
_class
loc:@pi/dense_1/kernel*
validate_shape(*
_output_shapes

:@@
┐
save_6/Assign_14Assignpi/dense_1/kernel/Adamsave_6/RestoreV2:14*$
_class
loc:@pi/dense_1/kernel*
validate_shape(*
_output_shapes

:@@*
use_locking(*
T0
┴
save_6/Assign_15Assignpi/dense_1/kernel/Adam_1save_6/RestoreV2:15*
T0*$
_class
loc:@pi/dense_1/kernel*
validate_shape(*
_output_shapes

:@@*
use_locking(
▓
save_6/Assign_16Assignpi/dense_2/biassave_6/RestoreV2:16*
validate_shape(*
_output_shapes
:*
use_locking(*
T0*"
_class
loc:@pi/dense_2/bias
и
save_6/Assign_17Assignpi/dense_2/bias/Adamsave_6/RestoreV2:17*
use_locking(*
T0*"
_class
loc:@pi/dense_2/bias*
validate_shape(*
_output_shapes
:
╣
save_6/Assign_18Assignpi/dense_2/bias/Adam_1save_6/RestoreV2:18*
use_locking(*
T0*"
_class
loc:@pi/dense_2/bias*
validate_shape(*
_output_shapes
:
║
save_6/Assign_19Assignpi/dense_2/kernelsave_6/RestoreV2:19*$
_class
loc:@pi/dense_2/kernel*
validate_shape(*
_output_shapes

:@*
use_locking(*
T0
┐
save_6/Assign_20Assignpi/dense_2/kernel/Adamsave_6/RestoreV2:20*$
_class
loc:@pi/dense_2/kernel*
validate_shape(*
_output_shapes

:@*
use_locking(*
T0
┴
save_6/Assign_21Assignpi/dense_2/kernel/Adam_1save_6/RestoreV2:21*
use_locking(*
T0*$
_class
loc:@pi/dense_2/kernel*
validate_shape(*
_output_shapes

:@
г
save_6/Assign_22Assignv/dense/biassave_6/RestoreV2:22*
use_locking(*
T0*
_class
loc:@v/dense/bias*
validate_shape(*
_output_shapes
:@
▒
save_6/Assign_23Assignv/dense/bias/Adamsave_6/RestoreV2:23*
T0*
_class
loc:@v/dense/bias*
validate_shape(*
_output_shapes
:@*
use_locking(
│
save_6/Assign_24Assignv/dense/bias/Adam_1save_6/RestoreV2:24*
use_locking(*
T0*
_class
loc:@v/dense/bias*
validate_shape(*
_output_shapes
:@
х
save_6/Assign_25Assignv/dense/kernelsave_6/RestoreV2:25*
validate_shape(*
_output_shapes
:	Є@*
use_locking(*
T0*!
_class
loc:@v/dense/kernel
║
save_6/Assign_26Assignv/dense/kernel/Adamsave_6/RestoreV2:26*
use_locking(*
T0*!
_class
loc:@v/dense/kernel*
validate_shape(*
_output_shapes
:	Є@
╝
save_6/Assign_27Assignv/dense/kernel/Adam_1save_6/RestoreV2:27*
_output_shapes
:	Є@*
use_locking(*
T0*!
_class
loc:@v/dense/kernel*
validate_shape(
░
save_6/Assign_28Assignv/dense_1/biassave_6/RestoreV2:28*
_output_shapes
:@*
use_locking(*
T0*!
_class
loc:@v/dense_1/bias*
validate_shape(
х
save_6/Assign_29Assignv/dense_1/bias/Adamsave_6/RestoreV2:29*
validate_shape(*
_output_shapes
:@*
use_locking(*
T0*!
_class
loc:@v/dense_1/bias
и
save_6/Assign_30Assignv/dense_1/bias/Adam_1save_6/RestoreV2:30*
use_locking(*
T0*!
_class
loc:@v/dense_1/bias*
validate_shape(*
_output_shapes
:@
И
save_6/Assign_31Assignv/dense_1/kernelsave_6/RestoreV2:31*
use_locking(*
T0*#
_class
loc:@v/dense_1/kernel*
validate_shape(*
_output_shapes

:@@
й
save_6/Assign_32Assignv/dense_1/kernel/Adamsave_6/RestoreV2:32*#
_class
loc:@v/dense_1/kernel*
validate_shape(*
_output_shapes

:@@*
use_locking(*
T0
┐
save_6/Assign_33Assignv/dense_1/kernel/Adam_1save_6/RestoreV2:33*
use_locking(*
T0*#
_class
loc:@v/dense_1/kernel*
validate_shape(*
_output_shapes

:@@
░
save_6/Assign_34Assignv/dense_2/biassave_6/RestoreV2:34*!
_class
loc:@v/dense_2/bias*
validate_shape(*
_output_shapes
:*
use_locking(*
T0
х
save_6/Assign_35Assignv/dense_2/bias/Adamsave_6/RestoreV2:35*
_output_shapes
:*
use_locking(*
T0*!
_class
loc:@v/dense_2/bias*
validate_shape(
и
save_6/Assign_36Assignv/dense_2/bias/Adam_1save_6/RestoreV2:36*
use_locking(*
T0*!
_class
loc:@v/dense_2/bias*
validate_shape(*
_output_shapes
:
И
save_6/Assign_37Assignv/dense_2/kernelsave_6/RestoreV2:37*
_output_shapes

:@*
use_locking(*
T0*#
_class
loc:@v/dense_2/kernel*
validate_shape(
й
save_6/Assign_38Assignv/dense_2/kernel/Adamsave_6/RestoreV2:38*
validate_shape(*
_output_shapes

:@*
use_locking(*
T0*#
_class
loc:@v/dense_2/kernel
┐
save_6/Assign_39Assignv/dense_2/kernel/Adam_1save_6/RestoreV2:39*
validate_shape(*
_output_shapes

:@*
use_locking(*
T0*#
_class
loc:@v/dense_2/kernel
ѕ
save_6/restore_shardNoOp^save_6/Assign^save_6/Assign_1^save_6/Assign_10^save_6/Assign_11^save_6/Assign_12^save_6/Assign_13^save_6/Assign_14^save_6/Assign_15^save_6/Assign_16^save_6/Assign_17^save_6/Assign_18^save_6/Assign_19^save_6/Assign_2^save_6/Assign_20^save_6/Assign_21^save_6/Assign_22^save_6/Assign_23^save_6/Assign_24^save_6/Assign_25^save_6/Assign_26^save_6/Assign_27^save_6/Assign_28^save_6/Assign_29^save_6/Assign_3^save_6/Assign_30^save_6/Assign_31^save_6/Assign_32^save_6/Assign_33^save_6/Assign_34^save_6/Assign_35^save_6/Assign_36^save_6/Assign_37^save_6/Assign_38^save_6/Assign_39^save_6/Assign_4^save_6/Assign_5^save_6/Assign_6^save_6/Assign_7^save_6/Assign_8^save_6/Assign_9
1
save_6/restore_allNoOp^save_6/restore_shard
[
save_7/filename/inputConst*
valueB Bmodel*
dtype0*
_output_shapes
: 
r
save_7/filenamePlaceholderWithDefaultsave_7/filename/input*
dtype0*
_output_shapes
: *
shape: 
i
save_7/ConstPlaceholderWithDefaultsave_7/filename*
dtype0*
_output_shapes
: *
shape: 
є
save_7/StringJoin/inputs_1Const*<
value3B1 B+_temp_9553fae7027549699b82ed201d2757c1/part*
dtype0*
_output_shapes
: 
{
save_7/StringJoin
StringJoinsave_7/Constsave_7/StringJoin/inputs_1*
_output_shapes
: *
	separator *
N
S
save_7/num_shardsConst*
value	B :*
dtype0*
_output_shapes
: 
^
save_7/ShardedFilename/shardConst*
value	B : *
dtype0*
_output_shapes
: 
Ё
save_7/ShardedFilenameShardedFilenamesave_7/StringJoinsave_7/ShardedFilename/shardsave_7/num_shards*
_output_shapes
: 
і
save_7/SaveV2/tensor_namesConst*╗
value▒B«(Bbeta1_powerBbeta1_power_1Bbeta2_powerBbeta2_power_1Bpi/dense/biasBpi/dense/bias/AdamBpi/dense/bias/Adam_1Bpi/dense/kernelBpi/dense/kernel/AdamBpi/dense/kernel/Adam_1Bpi/dense_1/biasBpi/dense_1/bias/AdamBpi/dense_1/bias/Adam_1Bpi/dense_1/kernelBpi/dense_1/kernel/AdamBpi/dense_1/kernel/Adam_1Bpi/dense_2/biasBpi/dense_2/bias/AdamBpi/dense_2/bias/Adam_1Bpi/dense_2/kernelBpi/dense_2/kernel/AdamBpi/dense_2/kernel/Adam_1Bv/dense/biasBv/dense/bias/AdamBv/dense/bias/Adam_1Bv/dense/kernelBv/dense/kernel/AdamBv/dense/kernel/Adam_1Bv/dense_1/biasBv/dense_1/bias/AdamBv/dense_1/bias/Adam_1Bv/dense_1/kernelBv/dense_1/kernel/AdamBv/dense_1/kernel/Adam_1Bv/dense_2/biasBv/dense_2/bias/AdamBv/dense_2/bias/Adam_1Bv/dense_2/kernelBv/dense_2/kernel/AdamBv/dense_2/kernel/Adam_1*
dtype0*
_output_shapes
:(
х
save_7/SaveV2/shape_and_slicesConst*c
valueZBX(B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B *
dtype0*
_output_shapes
:(
╔
save_7/SaveV2SaveV2save_7/ShardedFilenamesave_7/SaveV2/tensor_namessave_7/SaveV2/shape_and_slicesbeta1_powerbeta1_power_1beta2_powerbeta2_power_1pi/dense/biaspi/dense/bias/Adampi/dense/bias/Adam_1pi/dense/kernelpi/dense/kernel/Adampi/dense/kernel/Adam_1pi/dense_1/biaspi/dense_1/bias/Adampi/dense_1/bias/Adam_1pi/dense_1/kernelpi/dense_1/kernel/Adampi/dense_1/kernel/Adam_1pi/dense_2/biaspi/dense_2/bias/Adampi/dense_2/bias/Adam_1pi/dense_2/kernelpi/dense_2/kernel/Adampi/dense_2/kernel/Adam_1v/dense/biasv/dense/bias/Adamv/dense/bias/Adam_1v/dense/kernelv/dense/kernel/Adamv/dense/kernel/Adam_1v/dense_1/biasv/dense_1/bias/Adamv/dense_1/bias/Adam_1v/dense_1/kernelv/dense_1/kernel/Adamv/dense_1/kernel/Adam_1v/dense_2/biasv/dense_2/bias/Adamv/dense_2/bias/Adam_1v/dense_2/kernelv/dense_2/kernel/Adamv/dense_2/kernel/Adam_1*6
dtypes,
*2(
Ў
save_7/control_dependencyIdentitysave_7/ShardedFilename^save_7/SaveV2*
T0*)
_class
loc:@save_7/ShardedFilename*
_output_shapes
: 
Б
-save_7/MergeV2Checkpoints/checkpoint_prefixesPacksave_7/ShardedFilename^save_7/control_dependency*
T0*

axis *
N*
_output_shapes
:
Ѓ
save_7/MergeV2CheckpointsMergeV2Checkpoints-save_7/MergeV2Checkpoints/checkpoint_prefixessave_7/Const*
delete_old_dirs(
ѓ
save_7/IdentityIdentitysave_7/Const^save_7/MergeV2Checkpoints^save_7/control_dependency*
_output_shapes
: *
T0
Ї
save_7/RestoreV2/tensor_namesConst*╗
value▒B«(Bbeta1_powerBbeta1_power_1Bbeta2_powerBbeta2_power_1Bpi/dense/biasBpi/dense/bias/AdamBpi/dense/bias/Adam_1Bpi/dense/kernelBpi/dense/kernel/AdamBpi/dense/kernel/Adam_1Bpi/dense_1/biasBpi/dense_1/bias/AdamBpi/dense_1/bias/Adam_1Bpi/dense_1/kernelBpi/dense_1/kernel/AdamBpi/dense_1/kernel/Adam_1Bpi/dense_2/biasBpi/dense_2/bias/AdamBpi/dense_2/bias/Adam_1Bpi/dense_2/kernelBpi/dense_2/kernel/AdamBpi/dense_2/kernel/Adam_1Bv/dense/biasBv/dense/bias/AdamBv/dense/bias/Adam_1Bv/dense/kernelBv/dense/kernel/AdamBv/dense/kernel/Adam_1Bv/dense_1/biasBv/dense_1/bias/AdamBv/dense_1/bias/Adam_1Bv/dense_1/kernelBv/dense_1/kernel/AdamBv/dense_1/kernel/Adam_1Bv/dense_2/biasBv/dense_2/bias/AdamBv/dense_2/bias/Adam_1Bv/dense_2/kernelBv/dense_2/kernel/AdamBv/dense_2/kernel/Adam_1*
dtype0*
_output_shapes
:(
И
!save_7/RestoreV2/shape_and_slicesConst*c
valueZBX(B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B *
dtype0*
_output_shapes
:(
я
save_7/RestoreV2	RestoreV2save_7/Constsave_7/RestoreV2/tensor_names!save_7/RestoreV2/shape_and_slices*Х
_output_shapesБ
а::::::::::::::::::::::::::::::::::::::::*6
dtypes,
*2(
б
save_7/AssignAssignbeta1_powersave_7/RestoreV2*
validate_shape(*
_output_shapes
: *
use_locking(*
T0* 
_class
loc:@pi/dense/bias
Д
save_7/Assign_1Assignbeta1_power_1save_7/RestoreV2:1*
use_locking(*
T0*
_class
loc:@v/dense/bias*
validate_shape(*
_output_shapes
: 
д
save_7/Assign_2Assignbeta2_powersave_7/RestoreV2:2* 
_class
loc:@pi/dense/bias*
validate_shape(*
_output_shapes
: *
use_locking(*
T0
Д
save_7/Assign_3Assignbeta2_power_1save_7/RestoreV2:3*
use_locking(*
T0*
_class
loc:@v/dense/bias*
validate_shape(*
_output_shapes
: 
г
save_7/Assign_4Assignpi/dense/biassave_7/RestoreV2:4*
validate_shape(*
_output_shapes
:@*
use_locking(*
T0* 
_class
loc:@pi/dense/bias
▒
save_7/Assign_5Assignpi/dense/bias/Adamsave_7/RestoreV2:5*
validate_shape(*
_output_shapes
:@*
use_locking(*
T0* 
_class
loc:@pi/dense/bias
│
save_7/Assign_6Assignpi/dense/bias/Adam_1save_7/RestoreV2:6*
T0* 
_class
loc:@pi/dense/bias*
validate_shape(*
_output_shapes
:@*
use_locking(
х
save_7/Assign_7Assignpi/dense/kernelsave_7/RestoreV2:7*
validate_shape(*
_output_shapes
:	Є@*
use_locking(*
T0*"
_class
loc:@pi/dense/kernel
║
save_7/Assign_8Assignpi/dense/kernel/Adamsave_7/RestoreV2:8*
use_locking(*
T0*"
_class
loc:@pi/dense/kernel*
validate_shape(*
_output_shapes
:	Є@
╝
save_7/Assign_9Assignpi/dense/kernel/Adam_1save_7/RestoreV2:9*
use_locking(*
T0*"
_class
loc:@pi/dense/kernel*
validate_shape(*
_output_shapes
:	Є@
▓
save_7/Assign_10Assignpi/dense_1/biassave_7/RestoreV2:10*
validate_shape(*
_output_shapes
:@*
use_locking(*
T0*"
_class
loc:@pi/dense_1/bias
и
save_7/Assign_11Assignpi/dense_1/bias/Adamsave_7/RestoreV2:11*
use_locking(*
T0*"
_class
loc:@pi/dense_1/bias*
validate_shape(*
_output_shapes
:@
╣
save_7/Assign_12Assignpi/dense_1/bias/Adam_1save_7/RestoreV2:12*"
_class
loc:@pi/dense_1/bias*
validate_shape(*
_output_shapes
:@*
use_locking(*
T0
║
save_7/Assign_13Assignpi/dense_1/kernelsave_7/RestoreV2:13*
use_locking(*
T0*$
_class
loc:@pi/dense_1/kernel*
validate_shape(*
_output_shapes

:@@
┐
save_7/Assign_14Assignpi/dense_1/kernel/Adamsave_7/RestoreV2:14*
validate_shape(*
_output_shapes

:@@*
use_locking(*
T0*$
_class
loc:@pi/dense_1/kernel
┴
save_7/Assign_15Assignpi/dense_1/kernel/Adam_1save_7/RestoreV2:15*
_output_shapes

:@@*
use_locking(*
T0*$
_class
loc:@pi/dense_1/kernel*
validate_shape(
▓
save_7/Assign_16Assignpi/dense_2/biassave_7/RestoreV2:16*
validate_shape(*
_output_shapes
:*
use_locking(*
T0*"
_class
loc:@pi/dense_2/bias
и
save_7/Assign_17Assignpi/dense_2/bias/Adamsave_7/RestoreV2:17*
validate_shape(*
_output_shapes
:*
use_locking(*
T0*"
_class
loc:@pi/dense_2/bias
╣
save_7/Assign_18Assignpi/dense_2/bias/Adam_1save_7/RestoreV2:18*
use_locking(*
T0*"
_class
loc:@pi/dense_2/bias*
validate_shape(*
_output_shapes
:
║
save_7/Assign_19Assignpi/dense_2/kernelsave_7/RestoreV2:19*
T0*$
_class
loc:@pi/dense_2/kernel*
validate_shape(*
_output_shapes

:@*
use_locking(
┐
save_7/Assign_20Assignpi/dense_2/kernel/Adamsave_7/RestoreV2:20*
use_locking(*
T0*$
_class
loc:@pi/dense_2/kernel*
validate_shape(*
_output_shapes

:@
┴
save_7/Assign_21Assignpi/dense_2/kernel/Adam_1save_7/RestoreV2:21*
use_locking(*
T0*$
_class
loc:@pi/dense_2/kernel*
validate_shape(*
_output_shapes

:@
г
save_7/Assign_22Assignv/dense/biassave_7/RestoreV2:22*
T0*
_class
loc:@v/dense/bias*
validate_shape(*
_output_shapes
:@*
use_locking(
▒
save_7/Assign_23Assignv/dense/bias/Adamsave_7/RestoreV2:23*
use_locking(*
T0*
_class
loc:@v/dense/bias*
validate_shape(*
_output_shapes
:@
│
save_7/Assign_24Assignv/dense/bias/Adam_1save_7/RestoreV2:24*
_class
loc:@v/dense/bias*
validate_shape(*
_output_shapes
:@*
use_locking(*
T0
х
save_7/Assign_25Assignv/dense/kernelsave_7/RestoreV2:25*
T0*!
_class
loc:@v/dense/kernel*
validate_shape(*
_output_shapes
:	Є@*
use_locking(
║
save_7/Assign_26Assignv/dense/kernel/Adamsave_7/RestoreV2:26*
use_locking(*
T0*!
_class
loc:@v/dense/kernel*
validate_shape(*
_output_shapes
:	Є@
╝
save_7/Assign_27Assignv/dense/kernel/Adam_1save_7/RestoreV2:27*
use_locking(*
T0*!
_class
loc:@v/dense/kernel*
validate_shape(*
_output_shapes
:	Є@
░
save_7/Assign_28Assignv/dense_1/biassave_7/RestoreV2:28*
_output_shapes
:@*
use_locking(*
T0*!
_class
loc:@v/dense_1/bias*
validate_shape(
х
save_7/Assign_29Assignv/dense_1/bias/Adamsave_7/RestoreV2:29*
T0*!
_class
loc:@v/dense_1/bias*
validate_shape(*
_output_shapes
:@*
use_locking(
и
save_7/Assign_30Assignv/dense_1/bias/Adam_1save_7/RestoreV2:30*
validate_shape(*
_output_shapes
:@*
use_locking(*
T0*!
_class
loc:@v/dense_1/bias
И
save_7/Assign_31Assignv/dense_1/kernelsave_7/RestoreV2:31*
use_locking(*
T0*#
_class
loc:@v/dense_1/kernel*
validate_shape(*
_output_shapes

:@@
й
save_7/Assign_32Assignv/dense_1/kernel/Adamsave_7/RestoreV2:32*
use_locking(*
T0*#
_class
loc:@v/dense_1/kernel*
validate_shape(*
_output_shapes

:@@
┐
save_7/Assign_33Assignv/dense_1/kernel/Adam_1save_7/RestoreV2:33*
T0*#
_class
loc:@v/dense_1/kernel*
validate_shape(*
_output_shapes

:@@*
use_locking(
░
save_7/Assign_34Assignv/dense_2/biassave_7/RestoreV2:34*
validate_shape(*
_output_shapes
:*
use_locking(*
T0*!
_class
loc:@v/dense_2/bias
х
save_7/Assign_35Assignv/dense_2/bias/Adamsave_7/RestoreV2:35*!
_class
loc:@v/dense_2/bias*
validate_shape(*
_output_shapes
:*
use_locking(*
T0
и
save_7/Assign_36Assignv/dense_2/bias/Adam_1save_7/RestoreV2:36*
use_locking(*
T0*!
_class
loc:@v/dense_2/bias*
validate_shape(*
_output_shapes
:
И
save_7/Assign_37Assignv/dense_2/kernelsave_7/RestoreV2:37*
_output_shapes

:@*
use_locking(*
T0*#
_class
loc:@v/dense_2/kernel*
validate_shape(
й
save_7/Assign_38Assignv/dense_2/kernel/Adamsave_7/RestoreV2:38*
_output_shapes

:@*
use_locking(*
T0*#
_class
loc:@v/dense_2/kernel*
validate_shape(
┐
save_7/Assign_39Assignv/dense_2/kernel/Adam_1save_7/RestoreV2:39*
use_locking(*
T0*#
_class
loc:@v/dense_2/kernel*
validate_shape(*
_output_shapes

:@
ѕ
save_7/restore_shardNoOp^save_7/Assign^save_7/Assign_1^save_7/Assign_10^save_7/Assign_11^save_7/Assign_12^save_7/Assign_13^save_7/Assign_14^save_7/Assign_15^save_7/Assign_16^save_7/Assign_17^save_7/Assign_18^save_7/Assign_19^save_7/Assign_2^save_7/Assign_20^save_7/Assign_21^save_7/Assign_22^save_7/Assign_23^save_7/Assign_24^save_7/Assign_25^save_7/Assign_26^save_7/Assign_27^save_7/Assign_28^save_7/Assign_29^save_7/Assign_3^save_7/Assign_30^save_7/Assign_31^save_7/Assign_32^save_7/Assign_33^save_7/Assign_34^save_7/Assign_35^save_7/Assign_36^save_7/Assign_37^save_7/Assign_38^save_7/Assign_39^save_7/Assign_4^save_7/Assign_5^save_7/Assign_6^save_7/Assign_7^save_7/Assign_8^save_7/Assign_9
1
save_7/restore_allNoOp^save_7/restore_shard
[
save_8/filename/inputConst*
valueB Bmodel*
dtype0*
_output_shapes
: 
r
save_8/filenamePlaceholderWithDefaultsave_8/filename/input*
dtype0*
_output_shapes
: *
shape: 
i
save_8/ConstPlaceholderWithDefaultsave_8/filename*
_output_shapes
: *
shape: *
dtype0
є
save_8/StringJoin/inputs_1Const*<
value3B1 B+_temp_16178749a08a4a58bf26c0f8a18486d9/part*
dtype0*
_output_shapes
: 
{
save_8/StringJoin
StringJoinsave_8/Constsave_8/StringJoin/inputs_1*
N*
_output_shapes
: *
	separator 
S
save_8/num_shardsConst*
value	B :*
dtype0*
_output_shapes
: 
^
save_8/ShardedFilename/shardConst*
value	B : *
dtype0*
_output_shapes
: 
Ё
save_8/ShardedFilenameShardedFilenamesave_8/StringJoinsave_8/ShardedFilename/shardsave_8/num_shards*
_output_shapes
: 
і
save_8/SaveV2/tensor_namesConst*
_output_shapes
:(*╗
value▒B«(Bbeta1_powerBbeta1_power_1Bbeta2_powerBbeta2_power_1Bpi/dense/biasBpi/dense/bias/AdamBpi/dense/bias/Adam_1Bpi/dense/kernelBpi/dense/kernel/AdamBpi/dense/kernel/Adam_1Bpi/dense_1/biasBpi/dense_1/bias/AdamBpi/dense_1/bias/Adam_1Bpi/dense_1/kernelBpi/dense_1/kernel/AdamBpi/dense_1/kernel/Adam_1Bpi/dense_2/biasBpi/dense_2/bias/AdamBpi/dense_2/bias/Adam_1Bpi/dense_2/kernelBpi/dense_2/kernel/AdamBpi/dense_2/kernel/Adam_1Bv/dense/biasBv/dense/bias/AdamBv/dense/bias/Adam_1Bv/dense/kernelBv/dense/kernel/AdamBv/dense/kernel/Adam_1Bv/dense_1/biasBv/dense_1/bias/AdamBv/dense_1/bias/Adam_1Bv/dense_1/kernelBv/dense_1/kernel/AdamBv/dense_1/kernel/Adam_1Bv/dense_2/biasBv/dense_2/bias/AdamBv/dense_2/bias/Adam_1Bv/dense_2/kernelBv/dense_2/kernel/AdamBv/dense_2/kernel/Adam_1*
dtype0
х
save_8/SaveV2/shape_and_slicesConst*c
valueZBX(B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B *
dtype0*
_output_shapes
:(
╔
save_8/SaveV2SaveV2save_8/ShardedFilenamesave_8/SaveV2/tensor_namessave_8/SaveV2/shape_and_slicesbeta1_powerbeta1_power_1beta2_powerbeta2_power_1pi/dense/biaspi/dense/bias/Adampi/dense/bias/Adam_1pi/dense/kernelpi/dense/kernel/Adampi/dense/kernel/Adam_1pi/dense_1/biaspi/dense_1/bias/Adampi/dense_1/bias/Adam_1pi/dense_1/kernelpi/dense_1/kernel/Adampi/dense_1/kernel/Adam_1pi/dense_2/biaspi/dense_2/bias/Adampi/dense_2/bias/Adam_1pi/dense_2/kernelpi/dense_2/kernel/Adampi/dense_2/kernel/Adam_1v/dense/biasv/dense/bias/Adamv/dense/bias/Adam_1v/dense/kernelv/dense/kernel/Adamv/dense/kernel/Adam_1v/dense_1/biasv/dense_1/bias/Adamv/dense_1/bias/Adam_1v/dense_1/kernelv/dense_1/kernel/Adamv/dense_1/kernel/Adam_1v/dense_2/biasv/dense_2/bias/Adamv/dense_2/bias/Adam_1v/dense_2/kernelv/dense_2/kernel/Adamv/dense_2/kernel/Adam_1*6
dtypes,
*2(
Ў
save_8/control_dependencyIdentitysave_8/ShardedFilename^save_8/SaveV2*
_output_shapes
: *
T0*)
_class
loc:@save_8/ShardedFilename
Б
-save_8/MergeV2Checkpoints/checkpoint_prefixesPacksave_8/ShardedFilename^save_8/control_dependency*
T0*

axis *
N*
_output_shapes
:
Ѓ
save_8/MergeV2CheckpointsMergeV2Checkpoints-save_8/MergeV2Checkpoints/checkpoint_prefixessave_8/Const*
delete_old_dirs(
ѓ
save_8/IdentityIdentitysave_8/Const^save_8/MergeV2Checkpoints^save_8/control_dependency*
T0*
_output_shapes
: 
Ї
save_8/RestoreV2/tensor_namesConst*╗
value▒B«(Bbeta1_powerBbeta1_power_1Bbeta2_powerBbeta2_power_1Bpi/dense/biasBpi/dense/bias/AdamBpi/dense/bias/Adam_1Bpi/dense/kernelBpi/dense/kernel/AdamBpi/dense/kernel/Adam_1Bpi/dense_1/biasBpi/dense_1/bias/AdamBpi/dense_1/bias/Adam_1Bpi/dense_1/kernelBpi/dense_1/kernel/AdamBpi/dense_1/kernel/Adam_1Bpi/dense_2/biasBpi/dense_2/bias/AdamBpi/dense_2/bias/Adam_1Bpi/dense_2/kernelBpi/dense_2/kernel/AdamBpi/dense_2/kernel/Adam_1Bv/dense/biasBv/dense/bias/AdamBv/dense/bias/Adam_1Bv/dense/kernelBv/dense/kernel/AdamBv/dense/kernel/Adam_1Bv/dense_1/biasBv/dense_1/bias/AdamBv/dense_1/bias/Adam_1Bv/dense_1/kernelBv/dense_1/kernel/AdamBv/dense_1/kernel/Adam_1Bv/dense_2/biasBv/dense_2/bias/AdamBv/dense_2/bias/Adam_1Bv/dense_2/kernelBv/dense_2/kernel/AdamBv/dense_2/kernel/Adam_1*
dtype0*
_output_shapes
:(
И
!save_8/RestoreV2/shape_and_slicesConst*c
valueZBX(B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B *
dtype0*
_output_shapes
:(
я
save_8/RestoreV2	RestoreV2save_8/Constsave_8/RestoreV2/tensor_names!save_8/RestoreV2/shape_and_slices*6
dtypes,
*2(*Х
_output_shapesБ
а::::::::::::::::::::::::::::::::::::::::
б
save_8/AssignAssignbeta1_powersave_8/RestoreV2*
use_locking(*
T0* 
_class
loc:@pi/dense/bias*
validate_shape(*
_output_shapes
: 
Д
save_8/Assign_1Assignbeta1_power_1save_8/RestoreV2:1*
validate_shape(*
_output_shapes
: *
use_locking(*
T0*
_class
loc:@v/dense/bias
д
save_8/Assign_2Assignbeta2_powersave_8/RestoreV2:2*
use_locking(*
T0* 
_class
loc:@pi/dense/bias*
validate_shape(*
_output_shapes
: 
Д
save_8/Assign_3Assignbeta2_power_1save_8/RestoreV2:3*
use_locking(*
T0*
_class
loc:@v/dense/bias*
validate_shape(*
_output_shapes
: 
г
save_8/Assign_4Assignpi/dense/biassave_8/RestoreV2:4*
use_locking(*
T0* 
_class
loc:@pi/dense/bias*
validate_shape(*
_output_shapes
:@
▒
save_8/Assign_5Assignpi/dense/bias/Adamsave_8/RestoreV2:5*
use_locking(*
T0* 
_class
loc:@pi/dense/bias*
validate_shape(*
_output_shapes
:@
│
save_8/Assign_6Assignpi/dense/bias/Adam_1save_8/RestoreV2:6*
use_locking(*
T0* 
_class
loc:@pi/dense/bias*
validate_shape(*
_output_shapes
:@
х
save_8/Assign_7Assignpi/dense/kernelsave_8/RestoreV2:7*
use_locking(*
T0*"
_class
loc:@pi/dense/kernel*
validate_shape(*
_output_shapes
:	Є@
║
save_8/Assign_8Assignpi/dense/kernel/Adamsave_8/RestoreV2:8*
T0*"
_class
loc:@pi/dense/kernel*
validate_shape(*
_output_shapes
:	Є@*
use_locking(
╝
save_8/Assign_9Assignpi/dense/kernel/Adam_1save_8/RestoreV2:9*
use_locking(*
T0*"
_class
loc:@pi/dense/kernel*
validate_shape(*
_output_shapes
:	Є@
▓
save_8/Assign_10Assignpi/dense_1/biassave_8/RestoreV2:10*
T0*"
_class
loc:@pi/dense_1/bias*
validate_shape(*
_output_shapes
:@*
use_locking(
и
save_8/Assign_11Assignpi/dense_1/bias/Adamsave_8/RestoreV2:11*
validate_shape(*
_output_shapes
:@*
use_locking(*
T0*"
_class
loc:@pi/dense_1/bias
╣
save_8/Assign_12Assignpi/dense_1/bias/Adam_1save_8/RestoreV2:12*
use_locking(*
T0*"
_class
loc:@pi/dense_1/bias*
validate_shape(*
_output_shapes
:@
║
save_8/Assign_13Assignpi/dense_1/kernelsave_8/RestoreV2:13*$
_class
loc:@pi/dense_1/kernel*
validate_shape(*
_output_shapes

:@@*
use_locking(*
T0
┐
save_8/Assign_14Assignpi/dense_1/kernel/Adamsave_8/RestoreV2:14*$
_class
loc:@pi/dense_1/kernel*
validate_shape(*
_output_shapes

:@@*
use_locking(*
T0
┴
save_8/Assign_15Assignpi/dense_1/kernel/Adam_1save_8/RestoreV2:15*
use_locking(*
T0*$
_class
loc:@pi/dense_1/kernel*
validate_shape(*
_output_shapes

:@@
▓
save_8/Assign_16Assignpi/dense_2/biassave_8/RestoreV2:16*
use_locking(*
T0*"
_class
loc:@pi/dense_2/bias*
validate_shape(*
_output_shapes
:
и
save_8/Assign_17Assignpi/dense_2/bias/Adamsave_8/RestoreV2:17*
use_locking(*
T0*"
_class
loc:@pi/dense_2/bias*
validate_shape(*
_output_shapes
:
╣
save_8/Assign_18Assignpi/dense_2/bias/Adam_1save_8/RestoreV2:18*
use_locking(*
T0*"
_class
loc:@pi/dense_2/bias*
validate_shape(*
_output_shapes
:
║
save_8/Assign_19Assignpi/dense_2/kernelsave_8/RestoreV2:19*
_output_shapes

:@*
use_locking(*
T0*$
_class
loc:@pi/dense_2/kernel*
validate_shape(
┐
save_8/Assign_20Assignpi/dense_2/kernel/Adamsave_8/RestoreV2:20*
use_locking(*
T0*$
_class
loc:@pi/dense_2/kernel*
validate_shape(*
_output_shapes

:@
┴
save_8/Assign_21Assignpi/dense_2/kernel/Adam_1save_8/RestoreV2:21*
validate_shape(*
_output_shapes

:@*
use_locking(*
T0*$
_class
loc:@pi/dense_2/kernel
г
save_8/Assign_22Assignv/dense/biassave_8/RestoreV2:22*
use_locking(*
T0*
_class
loc:@v/dense/bias*
validate_shape(*
_output_shapes
:@
▒
save_8/Assign_23Assignv/dense/bias/Adamsave_8/RestoreV2:23*
_output_shapes
:@*
use_locking(*
T0*
_class
loc:@v/dense/bias*
validate_shape(
│
save_8/Assign_24Assignv/dense/bias/Adam_1save_8/RestoreV2:24*
T0*
_class
loc:@v/dense/bias*
validate_shape(*
_output_shapes
:@*
use_locking(
х
save_8/Assign_25Assignv/dense/kernelsave_8/RestoreV2:25*
use_locking(*
T0*!
_class
loc:@v/dense/kernel*
validate_shape(*
_output_shapes
:	Є@
║
save_8/Assign_26Assignv/dense/kernel/Adamsave_8/RestoreV2:26*
use_locking(*
T0*!
_class
loc:@v/dense/kernel*
validate_shape(*
_output_shapes
:	Є@
╝
save_8/Assign_27Assignv/dense/kernel/Adam_1save_8/RestoreV2:27*!
_class
loc:@v/dense/kernel*
validate_shape(*
_output_shapes
:	Є@*
use_locking(*
T0
░
save_8/Assign_28Assignv/dense_1/biassave_8/RestoreV2:28*
use_locking(*
T0*!
_class
loc:@v/dense_1/bias*
validate_shape(*
_output_shapes
:@
х
save_8/Assign_29Assignv/dense_1/bias/Adamsave_8/RestoreV2:29*
use_locking(*
T0*!
_class
loc:@v/dense_1/bias*
validate_shape(*
_output_shapes
:@
и
save_8/Assign_30Assignv/dense_1/bias/Adam_1save_8/RestoreV2:30*
T0*!
_class
loc:@v/dense_1/bias*
validate_shape(*
_output_shapes
:@*
use_locking(
И
save_8/Assign_31Assignv/dense_1/kernelsave_8/RestoreV2:31*
use_locking(*
T0*#
_class
loc:@v/dense_1/kernel*
validate_shape(*
_output_shapes

:@@
й
save_8/Assign_32Assignv/dense_1/kernel/Adamsave_8/RestoreV2:32*
validate_shape(*
_output_shapes

:@@*
use_locking(*
T0*#
_class
loc:@v/dense_1/kernel
┐
save_8/Assign_33Assignv/dense_1/kernel/Adam_1save_8/RestoreV2:33*#
_class
loc:@v/dense_1/kernel*
validate_shape(*
_output_shapes

:@@*
use_locking(*
T0
░
save_8/Assign_34Assignv/dense_2/biassave_8/RestoreV2:34*
use_locking(*
T0*!
_class
loc:@v/dense_2/bias*
validate_shape(*
_output_shapes
:
х
save_8/Assign_35Assignv/dense_2/bias/Adamsave_8/RestoreV2:35*
use_locking(*
T0*!
_class
loc:@v/dense_2/bias*
validate_shape(*
_output_shapes
:
и
save_8/Assign_36Assignv/dense_2/bias/Adam_1save_8/RestoreV2:36*
_output_shapes
:*
use_locking(*
T0*!
_class
loc:@v/dense_2/bias*
validate_shape(
И
save_8/Assign_37Assignv/dense_2/kernelsave_8/RestoreV2:37*
use_locking(*
T0*#
_class
loc:@v/dense_2/kernel*
validate_shape(*
_output_shapes

:@
й
save_8/Assign_38Assignv/dense_2/kernel/Adamsave_8/RestoreV2:38*
_output_shapes

:@*
use_locking(*
T0*#
_class
loc:@v/dense_2/kernel*
validate_shape(
┐
save_8/Assign_39Assignv/dense_2/kernel/Adam_1save_8/RestoreV2:39*
_output_shapes

:@*
use_locking(*
T0*#
_class
loc:@v/dense_2/kernel*
validate_shape(
ѕ
save_8/restore_shardNoOp^save_8/Assign^save_8/Assign_1^save_8/Assign_10^save_8/Assign_11^save_8/Assign_12^save_8/Assign_13^save_8/Assign_14^save_8/Assign_15^save_8/Assign_16^save_8/Assign_17^save_8/Assign_18^save_8/Assign_19^save_8/Assign_2^save_8/Assign_20^save_8/Assign_21^save_8/Assign_22^save_8/Assign_23^save_8/Assign_24^save_8/Assign_25^save_8/Assign_26^save_8/Assign_27^save_8/Assign_28^save_8/Assign_29^save_8/Assign_3^save_8/Assign_30^save_8/Assign_31^save_8/Assign_32^save_8/Assign_33^save_8/Assign_34^save_8/Assign_35^save_8/Assign_36^save_8/Assign_37^save_8/Assign_38^save_8/Assign_39^save_8/Assign_4^save_8/Assign_5^save_8/Assign_6^save_8/Assign_7^save_8/Assign_8^save_8/Assign_9
1
save_8/restore_allNoOp^save_8/restore_shard
[
save_9/filename/inputConst*
dtype0*
_output_shapes
: *
valueB Bmodel
r
save_9/filenamePlaceholderWithDefaultsave_9/filename/input*
dtype0*
_output_shapes
: *
shape: 
i
save_9/ConstPlaceholderWithDefaultsave_9/filename*
_output_shapes
: *
shape: *
dtype0
є
save_9/StringJoin/inputs_1Const*
dtype0*
_output_shapes
: *<
value3B1 B+_temp_a953091d68e44436b33c0749dac91d47/part
{
save_9/StringJoin
StringJoinsave_9/Constsave_9/StringJoin/inputs_1*
N*
_output_shapes
: *
	separator 
S
save_9/num_shardsConst*
value	B :*
dtype0*
_output_shapes
: 
^
save_9/ShardedFilename/shardConst*
value	B : *
dtype0*
_output_shapes
: 
Ё
save_9/ShardedFilenameShardedFilenamesave_9/StringJoinsave_9/ShardedFilename/shardsave_9/num_shards*
_output_shapes
: 
і
save_9/SaveV2/tensor_namesConst*╗
value▒B«(Bbeta1_powerBbeta1_power_1Bbeta2_powerBbeta2_power_1Bpi/dense/biasBpi/dense/bias/AdamBpi/dense/bias/Adam_1Bpi/dense/kernelBpi/dense/kernel/AdamBpi/dense/kernel/Adam_1Bpi/dense_1/biasBpi/dense_1/bias/AdamBpi/dense_1/bias/Adam_1Bpi/dense_1/kernelBpi/dense_1/kernel/AdamBpi/dense_1/kernel/Adam_1Bpi/dense_2/biasBpi/dense_2/bias/AdamBpi/dense_2/bias/Adam_1Bpi/dense_2/kernelBpi/dense_2/kernel/AdamBpi/dense_2/kernel/Adam_1Bv/dense/biasBv/dense/bias/AdamBv/dense/bias/Adam_1Bv/dense/kernelBv/dense/kernel/AdamBv/dense/kernel/Adam_1Bv/dense_1/biasBv/dense_1/bias/AdamBv/dense_1/bias/Adam_1Bv/dense_1/kernelBv/dense_1/kernel/AdamBv/dense_1/kernel/Adam_1Bv/dense_2/biasBv/dense_2/bias/AdamBv/dense_2/bias/Adam_1Bv/dense_2/kernelBv/dense_2/kernel/AdamBv/dense_2/kernel/Adam_1*
dtype0*
_output_shapes
:(
х
save_9/SaveV2/shape_and_slicesConst*c
valueZBX(B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B *
dtype0*
_output_shapes
:(
╔
save_9/SaveV2SaveV2save_9/ShardedFilenamesave_9/SaveV2/tensor_namessave_9/SaveV2/shape_and_slicesbeta1_powerbeta1_power_1beta2_powerbeta2_power_1pi/dense/biaspi/dense/bias/Adampi/dense/bias/Adam_1pi/dense/kernelpi/dense/kernel/Adampi/dense/kernel/Adam_1pi/dense_1/biaspi/dense_1/bias/Adampi/dense_1/bias/Adam_1pi/dense_1/kernelpi/dense_1/kernel/Adampi/dense_1/kernel/Adam_1pi/dense_2/biaspi/dense_2/bias/Adampi/dense_2/bias/Adam_1pi/dense_2/kernelpi/dense_2/kernel/Adampi/dense_2/kernel/Adam_1v/dense/biasv/dense/bias/Adamv/dense/bias/Adam_1v/dense/kernelv/dense/kernel/Adamv/dense/kernel/Adam_1v/dense_1/biasv/dense_1/bias/Adamv/dense_1/bias/Adam_1v/dense_1/kernelv/dense_1/kernel/Adamv/dense_1/kernel/Adam_1v/dense_2/biasv/dense_2/bias/Adamv/dense_2/bias/Adam_1v/dense_2/kernelv/dense_2/kernel/Adamv/dense_2/kernel/Adam_1*6
dtypes,
*2(
Ў
save_9/control_dependencyIdentitysave_9/ShardedFilename^save_9/SaveV2*
T0*)
_class
loc:@save_9/ShardedFilename*
_output_shapes
: 
Б
-save_9/MergeV2Checkpoints/checkpoint_prefixesPacksave_9/ShardedFilename^save_9/control_dependency*
N*
_output_shapes
:*
T0*

axis 
Ѓ
save_9/MergeV2CheckpointsMergeV2Checkpoints-save_9/MergeV2Checkpoints/checkpoint_prefixessave_9/Const*
delete_old_dirs(
ѓ
save_9/IdentityIdentitysave_9/Const^save_9/MergeV2Checkpoints^save_9/control_dependency*
T0*
_output_shapes
: 
Ї
save_9/RestoreV2/tensor_namesConst*╗
value▒B«(Bbeta1_powerBbeta1_power_1Bbeta2_powerBbeta2_power_1Bpi/dense/biasBpi/dense/bias/AdamBpi/dense/bias/Adam_1Bpi/dense/kernelBpi/dense/kernel/AdamBpi/dense/kernel/Adam_1Bpi/dense_1/biasBpi/dense_1/bias/AdamBpi/dense_1/bias/Adam_1Bpi/dense_1/kernelBpi/dense_1/kernel/AdamBpi/dense_1/kernel/Adam_1Bpi/dense_2/biasBpi/dense_2/bias/AdamBpi/dense_2/bias/Adam_1Bpi/dense_2/kernelBpi/dense_2/kernel/AdamBpi/dense_2/kernel/Adam_1Bv/dense/biasBv/dense/bias/AdamBv/dense/bias/Adam_1Bv/dense/kernelBv/dense/kernel/AdamBv/dense/kernel/Adam_1Bv/dense_1/biasBv/dense_1/bias/AdamBv/dense_1/bias/Adam_1Bv/dense_1/kernelBv/dense_1/kernel/AdamBv/dense_1/kernel/Adam_1Bv/dense_2/biasBv/dense_2/bias/AdamBv/dense_2/bias/Adam_1Bv/dense_2/kernelBv/dense_2/kernel/AdamBv/dense_2/kernel/Adam_1*
dtype0*
_output_shapes
:(
И
!save_9/RestoreV2/shape_and_slicesConst*c
valueZBX(B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B *
dtype0*
_output_shapes
:(
я
save_9/RestoreV2	RestoreV2save_9/Constsave_9/RestoreV2/tensor_names!save_9/RestoreV2/shape_and_slices*Х
_output_shapesБ
а::::::::::::::::::::::::::::::::::::::::*6
dtypes,
*2(
б
save_9/AssignAssignbeta1_powersave_9/RestoreV2*
T0* 
_class
loc:@pi/dense/bias*
validate_shape(*
_output_shapes
: *
use_locking(
Д
save_9/Assign_1Assignbeta1_power_1save_9/RestoreV2:1*
validate_shape(*
_output_shapes
: *
use_locking(*
T0*
_class
loc:@v/dense/bias
д
save_9/Assign_2Assignbeta2_powersave_9/RestoreV2:2*
use_locking(*
T0* 
_class
loc:@pi/dense/bias*
validate_shape(*
_output_shapes
: 
Д
save_9/Assign_3Assignbeta2_power_1save_9/RestoreV2:3*
use_locking(*
T0*
_class
loc:@v/dense/bias*
validate_shape(*
_output_shapes
: 
г
save_9/Assign_4Assignpi/dense/biassave_9/RestoreV2:4*
T0* 
_class
loc:@pi/dense/bias*
validate_shape(*
_output_shapes
:@*
use_locking(
▒
save_9/Assign_5Assignpi/dense/bias/Adamsave_9/RestoreV2:5*
use_locking(*
T0* 
_class
loc:@pi/dense/bias*
validate_shape(*
_output_shapes
:@
│
save_9/Assign_6Assignpi/dense/bias/Adam_1save_9/RestoreV2:6*
T0* 
_class
loc:@pi/dense/bias*
validate_shape(*
_output_shapes
:@*
use_locking(
х
save_9/Assign_7Assignpi/dense/kernelsave_9/RestoreV2:7*
_output_shapes
:	Є@*
use_locking(*
T0*"
_class
loc:@pi/dense/kernel*
validate_shape(
║
save_9/Assign_8Assignpi/dense/kernel/Adamsave_9/RestoreV2:8*
use_locking(*
T0*"
_class
loc:@pi/dense/kernel*
validate_shape(*
_output_shapes
:	Є@
╝
save_9/Assign_9Assignpi/dense/kernel/Adam_1save_9/RestoreV2:9*"
_class
loc:@pi/dense/kernel*
validate_shape(*
_output_shapes
:	Є@*
use_locking(*
T0
▓
save_9/Assign_10Assignpi/dense_1/biassave_9/RestoreV2:10*
use_locking(*
T0*"
_class
loc:@pi/dense_1/bias*
validate_shape(*
_output_shapes
:@
и
save_9/Assign_11Assignpi/dense_1/bias/Adamsave_9/RestoreV2:11*
validate_shape(*
_output_shapes
:@*
use_locking(*
T0*"
_class
loc:@pi/dense_1/bias
╣
save_9/Assign_12Assignpi/dense_1/bias/Adam_1save_9/RestoreV2:12*
use_locking(*
T0*"
_class
loc:@pi/dense_1/bias*
validate_shape(*
_output_shapes
:@
║
save_9/Assign_13Assignpi/dense_1/kernelsave_9/RestoreV2:13*
use_locking(*
T0*$
_class
loc:@pi/dense_1/kernel*
validate_shape(*
_output_shapes

:@@
┐
save_9/Assign_14Assignpi/dense_1/kernel/Adamsave_9/RestoreV2:14*
use_locking(*
T0*$
_class
loc:@pi/dense_1/kernel*
validate_shape(*
_output_shapes

:@@
┴
save_9/Assign_15Assignpi/dense_1/kernel/Adam_1save_9/RestoreV2:15*
validate_shape(*
_output_shapes

:@@*
use_locking(*
T0*$
_class
loc:@pi/dense_1/kernel
▓
save_9/Assign_16Assignpi/dense_2/biassave_9/RestoreV2:16*
use_locking(*
T0*"
_class
loc:@pi/dense_2/bias*
validate_shape(*
_output_shapes
:
и
save_9/Assign_17Assignpi/dense_2/bias/Adamsave_9/RestoreV2:17*"
_class
loc:@pi/dense_2/bias*
validate_shape(*
_output_shapes
:*
use_locking(*
T0
╣
save_9/Assign_18Assignpi/dense_2/bias/Adam_1save_9/RestoreV2:18*
use_locking(*
T0*"
_class
loc:@pi/dense_2/bias*
validate_shape(*
_output_shapes
:
║
save_9/Assign_19Assignpi/dense_2/kernelsave_9/RestoreV2:19*
use_locking(*
T0*$
_class
loc:@pi/dense_2/kernel*
validate_shape(*
_output_shapes

:@
┐
save_9/Assign_20Assignpi/dense_2/kernel/Adamsave_9/RestoreV2:20*
use_locking(*
T0*$
_class
loc:@pi/dense_2/kernel*
validate_shape(*
_output_shapes

:@
┴
save_9/Assign_21Assignpi/dense_2/kernel/Adam_1save_9/RestoreV2:21*$
_class
loc:@pi/dense_2/kernel*
validate_shape(*
_output_shapes

:@*
use_locking(*
T0
г
save_9/Assign_22Assignv/dense/biassave_9/RestoreV2:22*
T0*
_class
loc:@v/dense/bias*
validate_shape(*
_output_shapes
:@*
use_locking(
▒
save_9/Assign_23Assignv/dense/bias/Adamsave_9/RestoreV2:23*
use_locking(*
T0*
_class
loc:@v/dense/bias*
validate_shape(*
_output_shapes
:@
│
save_9/Assign_24Assignv/dense/bias/Adam_1save_9/RestoreV2:24*
use_locking(*
T0*
_class
loc:@v/dense/bias*
validate_shape(*
_output_shapes
:@
х
save_9/Assign_25Assignv/dense/kernelsave_9/RestoreV2:25*
use_locking(*
T0*!
_class
loc:@v/dense/kernel*
validate_shape(*
_output_shapes
:	Є@
║
save_9/Assign_26Assignv/dense/kernel/Adamsave_9/RestoreV2:26*
_output_shapes
:	Є@*
use_locking(*
T0*!
_class
loc:@v/dense/kernel*
validate_shape(
╝
save_9/Assign_27Assignv/dense/kernel/Adam_1save_9/RestoreV2:27*
validate_shape(*
_output_shapes
:	Є@*
use_locking(*
T0*!
_class
loc:@v/dense/kernel
░
save_9/Assign_28Assignv/dense_1/biassave_9/RestoreV2:28*
validate_shape(*
_output_shapes
:@*
use_locking(*
T0*!
_class
loc:@v/dense_1/bias
х
save_9/Assign_29Assignv/dense_1/bias/Adamsave_9/RestoreV2:29*
use_locking(*
T0*!
_class
loc:@v/dense_1/bias*
validate_shape(*
_output_shapes
:@
и
save_9/Assign_30Assignv/dense_1/bias/Adam_1save_9/RestoreV2:30*
_output_shapes
:@*
use_locking(*
T0*!
_class
loc:@v/dense_1/bias*
validate_shape(
И
save_9/Assign_31Assignv/dense_1/kernelsave_9/RestoreV2:31*
_output_shapes

:@@*
use_locking(*
T0*#
_class
loc:@v/dense_1/kernel*
validate_shape(
й
save_9/Assign_32Assignv/dense_1/kernel/Adamsave_9/RestoreV2:32*
use_locking(*
T0*#
_class
loc:@v/dense_1/kernel*
validate_shape(*
_output_shapes

:@@
┐
save_9/Assign_33Assignv/dense_1/kernel/Adam_1save_9/RestoreV2:33*#
_class
loc:@v/dense_1/kernel*
validate_shape(*
_output_shapes

:@@*
use_locking(*
T0
░
save_9/Assign_34Assignv/dense_2/biassave_9/RestoreV2:34*
use_locking(*
T0*!
_class
loc:@v/dense_2/bias*
validate_shape(*
_output_shapes
:
х
save_9/Assign_35Assignv/dense_2/bias/Adamsave_9/RestoreV2:35*
validate_shape(*
_output_shapes
:*
use_locking(*
T0*!
_class
loc:@v/dense_2/bias
и
save_9/Assign_36Assignv/dense_2/bias/Adam_1save_9/RestoreV2:36*
use_locking(*
T0*!
_class
loc:@v/dense_2/bias*
validate_shape(*
_output_shapes
:
И
save_9/Assign_37Assignv/dense_2/kernelsave_9/RestoreV2:37*
use_locking(*
T0*#
_class
loc:@v/dense_2/kernel*
validate_shape(*
_output_shapes

:@
й
save_9/Assign_38Assignv/dense_2/kernel/Adamsave_9/RestoreV2:38*
_output_shapes

:@*
use_locking(*
T0*#
_class
loc:@v/dense_2/kernel*
validate_shape(
┐
save_9/Assign_39Assignv/dense_2/kernel/Adam_1save_9/RestoreV2:39*
T0*#
_class
loc:@v/dense_2/kernel*
validate_shape(*
_output_shapes

:@*
use_locking(
ѕ
save_9/restore_shardNoOp^save_9/Assign^save_9/Assign_1^save_9/Assign_10^save_9/Assign_11^save_9/Assign_12^save_9/Assign_13^save_9/Assign_14^save_9/Assign_15^save_9/Assign_16^save_9/Assign_17^save_9/Assign_18^save_9/Assign_19^save_9/Assign_2^save_9/Assign_20^save_9/Assign_21^save_9/Assign_22^save_9/Assign_23^save_9/Assign_24^save_9/Assign_25^save_9/Assign_26^save_9/Assign_27^save_9/Assign_28^save_9/Assign_29^save_9/Assign_3^save_9/Assign_30^save_9/Assign_31^save_9/Assign_32^save_9/Assign_33^save_9/Assign_34^save_9/Assign_35^save_9/Assign_36^save_9/Assign_37^save_9/Assign_38^save_9/Assign_39^save_9/Assign_4^save_9/Assign_5^save_9/Assign_6^save_9/Assign_7^save_9/Assign_8^save_9/Assign_9
1
save_9/restore_allNoOp^save_9/restore_shard
\
save_10/filename/inputConst*
valueB Bmodel*
dtype0*
_output_shapes
: 
t
save_10/filenamePlaceholderWithDefaultsave_10/filename/input*
dtype0*
_output_shapes
: *
shape: 
k
save_10/ConstPlaceholderWithDefaultsave_10/filename*
_output_shapes
: *
shape: *
dtype0
Є
save_10/StringJoin/inputs_1Const*<
value3B1 B+_temp_d7c7de99c848472093a21c3a8cea72f2/part*
dtype0*
_output_shapes
: 
~
save_10/StringJoin
StringJoinsave_10/Constsave_10/StringJoin/inputs_1*
	separator *
N*
_output_shapes
: 
T
save_10/num_shardsConst*
value	B :*
dtype0*
_output_shapes
: 
_
save_10/ShardedFilename/shardConst*
value	B : *
dtype0*
_output_shapes
: 
Ѕ
save_10/ShardedFilenameShardedFilenamesave_10/StringJoinsave_10/ShardedFilename/shardsave_10/num_shards*
_output_shapes
: 
І
save_10/SaveV2/tensor_namesConst*
_output_shapes
:(*╗
value▒B«(Bbeta1_powerBbeta1_power_1Bbeta2_powerBbeta2_power_1Bpi/dense/biasBpi/dense/bias/AdamBpi/dense/bias/Adam_1Bpi/dense/kernelBpi/dense/kernel/AdamBpi/dense/kernel/Adam_1Bpi/dense_1/biasBpi/dense_1/bias/AdamBpi/dense_1/bias/Adam_1Bpi/dense_1/kernelBpi/dense_1/kernel/AdamBpi/dense_1/kernel/Adam_1Bpi/dense_2/biasBpi/dense_2/bias/AdamBpi/dense_2/bias/Adam_1Bpi/dense_2/kernelBpi/dense_2/kernel/AdamBpi/dense_2/kernel/Adam_1Bv/dense/biasBv/dense/bias/AdamBv/dense/bias/Adam_1Bv/dense/kernelBv/dense/kernel/AdamBv/dense/kernel/Adam_1Bv/dense_1/biasBv/dense_1/bias/AdamBv/dense_1/bias/Adam_1Bv/dense_1/kernelBv/dense_1/kernel/AdamBv/dense_1/kernel/Adam_1Bv/dense_2/biasBv/dense_2/bias/AdamBv/dense_2/bias/Adam_1Bv/dense_2/kernelBv/dense_2/kernel/AdamBv/dense_2/kernel/Adam_1*
dtype0
Х
save_10/SaveV2/shape_and_slicesConst*
dtype0*
_output_shapes
:(*c
valueZBX(B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 
═
save_10/SaveV2SaveV2save_10/ShardedFilenamesave_10/SaveV2/tensor_namessave_10/SaveV2/shape_and_slicesbeta1_powerbeta1_power_1beta2_powerbeta2_power_1pi/dense/biaspi/dense/bias/Adampi/dense/bias/Adam_1pi/dense/kernelpi/dense/kernel/Adampi/dense/kernel/Adam_1pi/dense_1/biaspi/dense_1/bias/Adampi/dense_1/bias/Adam_1pi/dense_1/kernelpi/dense_1/kernel/Adampi/dense_1/kernel/Adam_1pi/dense_2/biaspi/dense_2/bias/Adampi/dense_2/bias/Adam_1pi/dense_2/kernelpi/dense_2/kernel/Adampi/dense_2/kernel/Adam_1v/dense/biasv/dense/bias/Adamv/dense/bias/Adam_1v/dense/kernelv/dense/kernel/Adamv/dense/kernel/Adam_1v/dense_1/biasv/dense_1/bias/Adamv/dense_1/bias/Adam_1v/dense_1/kernelv/dense_1/kernel/Adamv/dense_1/kernel/Adam_1v/dense_2/biasv/dense_2/bias/Adamv/dense_2/bias/Adam_1v/dense_2/kernelv/dense_2/kernel/Adamv/dense_2/kernel/Adam_1*6
dtypes,
*2(
Ю
save_10/control_dependencyIdentitysave_10/ShardedFilename^save_10/SaveV2*
T0**
_class 
loc:@save_10/ShardedFilename*
_output_shapes
: 
д
.save_10/MergeV2Checkpoints/checkpoint_prefixesPacksave_10/ShardedFilename^save_10/control_dependency*
T0*

axis *
N*
_output_shapes
:
є
save_10/MergeV2CheckpointsMergeV2Checkpoints.save_10/MergeV2Checkpoints/checkpoint_prefixessave_10/Const*
delete_old_dirs(
є
save_10/IdentityIdentitysave_10/Const^save_10/MergeV2Checkpoints^save_10/control_dependency*
_output_shapes
: *
T0
ј
save_10/RestoreV2/tensor_namesConst*╗
value▒B«(Bbeta1_powerBbeta1_power_1Bbeta2_powerBbeta2_power_1Bpi/dense/biasBpi/dense/bias/AdamBpi/dense/bias/Adam_1Bpi/dense/kernelBpi/dense/kernel/AdamBpi/dense/kernel/Adam_1Bpi/dense_1/biasBpi/dense_1/bias/AdamBpi/dense_1/bias/Adam_1Bpi/dense_1/kernelBpi/dense_1/kernel/AdamBpi/dense_1/kernel/Adam_1Bpi/dense_2/biasBpi/dense_2/bias/AdamBpi/dense_2/bias/Adam_1Bpi/dense_2/kernelBpi/dense_2/kernel/AdamBpi/dense_2/kernel/Adam_1Bv/dense/biasBv/dense/bias/AdamBv/dense/bias/Adam_1Bv/dense/kernelBv/dense/kernel/AdamBv/dense/kernel/Adam_1Bv/dense_1/biasBv/dense_1/bias/AdamBv/dense_1/bias/Adam_1Bv/dense_1/kernelBv/dense_1/kernel/AdamBv/dense_1/kernel/Adam_1Bv/dense_2/biasBv/dense_2/bias/AdamBv/dense_2/bias/Adam_1Bv/dense_2/kernelBv/dense_2/kernel/AdamBv/dense_2/kernel/Adam_1*
dtype0*
_output_shapes
:(
╣
"save_10/RestoreV2/shape_and_slicesConst*c
valueZBX(B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B *
dtype0*
_output_shapes
:(
Р
save_10/RestoreV2	RestoreV2save_10/Constsave_10/RestoreV2/tensor_names"save_10/RestoreV2/shape_and_slices*Х
_output_shapesБ
а::::::::::::::::::::::::::::::::::::::::*6
dtypes,
*2(
ц
save_10/AssignAssignbeta1_powersave_10/RestoreV2*
use_locking(*
T0* 
_class
loc:@pi/dense/bias*
validate_shape(*
_output_shapes
: 
Е
save_10/Assign_1Assignbeta1_power_1save_10/RestoreV2:1*
_output_shapes
: *
use_locking(*
T0*
_class
loc:@v/dense/bias*
validate_shape(
е
save_10/Assign_2Assignbeta2_powersave_10/RestoreV2:2*
T0* 
_class
loc:@pi/dense/bias*
validate_shape(*
_output_shapes
: *
use_locking(
Е
save_10/Assign_3Assignbeta2_power_1save_10/RestoreV2:3*
use_locking(*
T0*
_class
loc:@v/dense/bias*
validate_shape(*
_output_shapes
: 
«
save_10/Assign_4Assignpi/dense/biassave_10/RestoreV2:4*
T0* 
_class
loc:@pi/dense/bias*
validate_shape(*
_output_shapes
:@*
use_locking(
│
save_10/Assign_5Assignpi/dense/bias/Adamsave_10/RestoreV2:5*
use_locking(*
T0* 
_class
loc:@pi/dense/bias*
validate_shape(*
_output_shapes
:@
х
save_10/Assign_6Assignpi/dense/bias/Adam_1save_10/RestoreV2:6*
use_locking(*
T0* 
_class
loc:@pi/dense/bias*
validate_shape(*
_output_shapes
:@
и
save_10/Assign_7Assignpi/dense/kernelsave_10/RestoreV2:7*
T0*"
_class
loc:@pi/dense/kernel*
validate_shape(*
_output_shapes
:	Є@*
use_locking(
╝
save_10/Assign_8Assignpi/dense/kernel/Adamsave_10/RestoreV2:8*
use_locking(*
T0*"
_class
loc:@pi/dense/kernel*
validate_shape(*
_output_shapes
:	Є@
Й
save_10/Assign_9Assignpi/dense/kernel/Adam_1save_10/RestoreV2:9*
use_locking(*
T0*"
_class
loc:@pi/dense/kernel*
validate_shape(*
_output_shapes
:	Є@
┤
save_10/Assign_10Assignpi/dense_1/biassave_10/RestoreV2:10*
validate_shape(*
_output_shapes
:@*
use_locking(*
T0*"
_class
loc:@pi/dense_1/bias
╣
save_10/Assign_11Assignpi/dense_1/bias/Adamsave_10/RestoreV2:11*
_output_shapes
:@*
use_locking(*
T0*"
_class
loc:@pi/dense_1/bias*
validate_shape(
╗
save_10/Assign_12Assignpi/dense_1/bias/Adam_1save_10/RestoreV2:12*
use_locking(*
T0*"
_class
loc:@pi/dense_1/bias*
validate_shape(*
_output_shapes
:@
╝
save_10/Assign_13Assignpi/dense_1/kernelsave_10/RestoreV2:13*
use_locking(*
T0*$
_class
loc:@pi/dense_1/kernel*
validate_shape(*
_output_shapes

:@@
┴
save_10/Assign_14Assignpi/dense_1/kernel/Adamsave_10/RestoreV2:14*
use_locking(*
T0*$
_class
loc:@pi/dense_1/kernel*
validate_shape(*
_output_shapes

:@@
├
save_10/Assign_15Assignpi/dense_1/kernel/Adam_1save_10/RestoreV2:15*
use_locking(*
T0*$
_class
loc:@pi/dense_1/kernel*
validate_shape(*
_output_shapes

:@@
┤
save_10/Assign_16Assignpi/dense_2/biassave_10/RestoreV2:16*
use_locking(*
T0*"
_class
loc:@pi/dense_2/bias*
validate_shape(*
_output_shapes
:
╣
save_10/Assign_17Assignpi/dense_2/bias/Adamsave_10/RestoreV2:17*
use_locking(*
T0*"
_class
loc:@pi/dense_2/bias*
validate_shape(*
_output_shapes
:
╗
save_10/Assign_18Assignpi/dense_2/bias/Adam_1save_10/RestoreV2:18*
T0*"
_class
loc:@pi/dense_2/bias*
validate_shape(*
_output_shapes
:*
use_locking(
╝
save_10/Assign_19Assignpi/dense_2/kernelsave_10/RestoreV2:19*
T0*$
_class
loc:@pi/dense_2/kernel*
validate_shape(*
_output_shapes

:@*
use_locking(
┴
save_10/Assign_20Assignpi/dense_2/kernel/Adamsave_10/RestoreV2:20*
T0*$
_class
loc:@pi/dense_2/kernel*
validate_shape(*
_output_shapes

:@*
use_locking(
├
save_10/Assign_21Assignpi/dense_2/kernel/Adam_1save_10/RestoreV2:21*
use_locking(*
T0*$
_class
loc:@pi/dense_2/kernel*
validate_shape(*
_output_shapes

:@
«
save_10/Assign_22Assignv/dense/biassave_10/RestoreV2:22*
use_locking(*
T0*
_class
loc:@v/dense/bias*
validate_shape(*
_output_shapes
:@
│
save_10/Assign_23Assignv/dense/bias/Adamsave_10/RestoreV2:23*
_output_shapes
:@*
use_locking(*
T0*
_class
loc:@v/dense/bias*
validate_shape(
х
save_10/Assign_24Assignv/dense/bias/Adam_1save_10/RestoreV2:24*
use_locking(*
T0*
_class
loc:@v/dense/bias*
validate_shape(*
_output_shapes
:@
и
save_10/Assign_25Assignv/dense/kernelsave_10/RestoreV2:25*
_output_shapes
:	Є@*
use_locking(*
T0*!
_class
loc:@v/dense/kernel*
validate_shape(
╝
save_10/Assign_26Assignv/dense/kernel/Adamsave_10/RestoreV2:26*
use_locking(*
T0*!
_class
loc:@v/dense/kernel*
validate_shape(*
_output_shapes
:	Є@
Й
save_10/Assign_27Assignv/dense/kernel/Adam_1save_10/RestoreV2:27*
use_locking(*
T0*!
_class
loc:@v/dense/kernel*
validate_shape(*
_output_shapes
:	Є@
▓
save_10/Assign_28Assignv/dense_1/biassave_10/RestoreV2:28*
use_locking(*
T0*!
_class
loc:@v/dense_1/bias*
validate_shape(*
_output_shapes
:@
и
save_10/Assign_29Assignv/dense_1/bias/Adamsave_10/RestoreV2:29*
use_locking(*
T0*!
_class
loc:@v/dense_1/bias*
validate_shape(*
_output_shapes
:@
╣
save_10/Assign_30Assignv/dense_1/bias/Adam_1save_10/RestoreV2:30*
use_locking(*
T0*!
_class
loc:@v/dense_1/bias*
validate_shape(*
_output_shapes
:@
║
save_10/Assign_31Assignv/dense_1/kernelsave_10/RestoreV2:31*
use_locking(*
T0*#
_class
loc:@v/dense_1/kernel*
validate_shape(*
_output_shapes

:@@
┐
save_10/Assign_32Assignv/dense_1/kernel/Adamsave_10/RestoreV2:32*
use_locking(*
T0*#
_class
loc:@v/dense_1/kernel*
validate_shape(*
_output_shapes

:@@
┴
save_10/Assign_33Assignv/dense_1/kernel/Adam_1save_10/RestoreV2:33*
_output_shapes

:@@*
use_locking(*
T0*#
_class
loc:@v/dense_1/kernel*
validate_shape(
▓
save_10/Assign_34Assignv/dense_2/biassave_10/RestoreV2:34*
_output_shapes
:*
use_locking(*
T0*!
_class
loc:@v/dense_2/bias*
validate_shape(
и
save_10/Assign_35Assignv/dense_2/bias/Adamsave_10/RestoreV2:35*
T0*!
_class
loc:@v/dense_2/bias*
validate_shape(*
_output_shapes
:*
use_locking(
╣
save_10/Assign_36Assignv/dense_2/bias/Adam_1save_10/RestoreV2:36*
T0*!
_class
loc:@v/dense_2/bias*
validate_shape(*
_output_shapes
:*
use_locking(
║
save_10/Assign_37Assignv/dense_2/kernelsave_10/RestoreV2:37*
use_locking(*
T0*#
_class
loc:@v/dense_2/kernel*
validate_shape(*
_output_shapes

:@
┐
save_10/Assign_38Assignv/dense_2/kernel/Adamsave_10/RestoreV2:38*
use_locking(*
T0*#
_class
loc:@v/dense_2/kernel*
validate_shape(*
_output_shapes

:@
┴
save_10/Assign_39Assignv/dense_2/kernel/Adam_1save_10/RestoreV2:39*
use_locking(*
T0*#
_class
loc:@v/dense_2/kernel*
validate_shape(*
_output_shapes

:@
▒
save_10/restore_shardNoOp^save_10/Assign^save_10/Assign_1^save_10/Assign_10^save_10/Assign_11^save_10/Assign_12^save_10/Assign_13^save_10/Assign_14^save_10/Assign_15^save_10/Assign_16^save_10/Assign_17^save_10/Assign_18^save_10/Assign_19^save_10/Assign_2^save_10/Assign_20^save_10/Assign_21^save_10/Assign_22^save_10/Assign_23^save_10/Assign_24^save_10/Assign_25^save_10/Assign_26^save_10/Assign_27^save_10/Assign_28^save_10/Assign_29^save_10/Assign_3^save_10/Assign_30^save_10/Assign_31^save_10/Assign_32^save_10/Assign_33^save_10/Assign_34^save_10/Assign_35^save_10/Assign_36^save_10/Assign_37^save_10/Assign_38^save_10/Assign_39^save_10/Assign_4^save_10/Assign_5^save_10/Assign_6^save_10/Assign_7^save_10/Assign_8^save_10/Assign_9
3
save_10/restore_allNoOp^save_10/restore_shard
\
save_11/filename/inputConst*
_output_shapes
: *
valueB Bmodel*
dtype0
t
save_11/filenamePlaceholderWithDefaultsave_11/filename/input*
dtype0*
_output_shapes
: *
shape: 
k
save_11/ConstPlaceholderWithDefaultsave_11/filename*
dtype0*
_output_shapes
: *
shape: 
Є
save_11/StringJoin/inputs_1Const*
dtype0*
_output_shapes
: *<
value3B1 B+_temp_4a20d20cb7644529ba6b658a2cb9b252/part
~
save_11/StringJoin
StringJoinsave_11/Constsave_11/StringJoin/inputs_1*
N*
_output_shapes
: *
	separator 
T
save_11/num_shardsConst*
value	B :*
dtype0*
_output_shapes
: 
_
save_11/ShardedFilename/shardConst*
value	B : *
dtype0*
_output_shapes
: 
Ѕ
save_11/ShardedFilenameShardedFilenamesave_11/StringJoinsave_11/ShardedFilename/shardsave_11/num_shards*
_output_shapes
: 
І
save_11/SaveV2/tensor_namesConst*╗
value▒B«(Bbeta1_powerBbeta1_power_1Bbeta2_powerBbeta2_power_1Bpi/dense/biasBpi/dense/bias/AdamBpi/dense/bias/Adam_1Bpi/dense/kernelBpi/dense/kernel/AdamBpi/dense/kernel/Adam_1Bpi/dense_1/biasBpi/dense_1/bias/AdamBpi/dense_1/bias/Adam_1Bpi/dense_1/kernelBpi/dense_1/kernel/AdamBpi/dense_1/kernel/Adam_1Bpi/dense_2/biasBpi/dense_2/bias/AdamBpi/dense_2/bias/Adam_1Bpi/dense_2/kernelBpi/dense_2/kernel/AdamBpi/dense_2/kernel/Adam_1Bv/dense/biasBv/dense/bias/AdamBv/dense/bias/Adam_1Bv/dense/kernelBv/dense/kernel/AdamBv/dense/kernel/Adam_1Bv/dense_1/biasBv/dense_1/bias/AdamBv/dense_1/bias/Adam_1Bv/dense_1/kernelBv/dense_1/kernel/AdamBv/dense_1/kernel/Adam_1Bv/dense_2/biasBv/dense_2/bias/AdamBv/dense_2/bias/Adam_1Bv/dense_2/kernelBv/dense_2/kernel/AdamBv/dense_2/kernel/Adam_1*
dtype0*
_output_shapes
:(
Х
save_11/SaveV2/shape_and_slicesConst*
_output_shapes
:(*c
valueZBX(B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B *
dtype0
═
save_11/SaveV2SaveV2save_11/ShardedFilenamesave_11/SaveV2/tensor_namessave_11/SaveV2/shape_and_slicesbeta1_powerbeta1_power_1beta2_powerbeta2_power_1pi/dense/biaspi/dense/bias/Adampi/dense/bias/Adam_1pi/dense/kernelpi/dense/kernel/Adampi/dense/kernel/Adam_1pi/dense_1/biaspi/dense_1/bias/Adampi/dense_1/bias/Adam_1pi/dense_1/kernelpi/dense_1/kernel/Adampi/dense_1/kernel/Adam_1pi/dense_2/biaspi/dense_2/bias/Adampi/dense_2/bias/Adam_1pi/dense_2/kernelpi/dense_2/kernel/Adampi/dense_2/kernel/Adam_1v/dense/biasv/dense/bias/Adamv/dense/bias/Adam_1v/dense/kernelv/dense/kernel/Adamv/dense/kernel/Adam_1v/dense_1/biasv/dense_1/bias/Adamv/dense_1/bias/Adam_1v/dense_1/kernelv/dense_1/kernel/Adamv/dense_1/kernel/Adam_1v/dense_2/biasv/dense_2/bias/Adamv/dense_2/bias/Adam_1v/dense_2/kernelv/dense_2/kernel/Adamv/dense_2/kernel/Adam_1*6
dtypes,
*2(
Ю
save_11/control_dependencyIdentitysave_11/ShardedFilename^save_11/SaveV2*
_output_shapes
: *
T0**
_class 
loc:@save_11/ShardedFilename
д
.save_11/MergeV2Checkpoints/checkpoint_prefixesPacksave_11/ShardedFilename^save_11/control_dependency*
_output_shapes
:*
T0*

axis *
N
є
save_11/MergeV2CheckpointsMergeV2Checkpoints.save_11/MergeV2Checkpoints/checkpoint_prefixessave_11/Const*
delete_old_dirs(
є
save_11/IdentityIdentitysave_11/Const^save_11/MergeV2Checkpoints^save_11/control_dependency*
_output_shapes
: *
T0
ј
save_11/RestoreV2/tensor_namesConst*╗
value▒B«(Bbeta1_powerBbeta1_power_1Bbeta2_powerBbeta2_power_1Bpi/dense/biasBpi/dense/bias/AdamBpi/dense/bias/Adam_1Bpi/dense/kernelBpi/dense/kernel/AdamBpi/dense/kernel/Adam_1Bpi/dense_1/biasBpi/dense_1/bias/AdamBpi/dense_1/bias/Adam_1Bpi/dense_1/kernelBpi/dense_1/kernel/AdamBpi/dense_1/kernel/Adam_1Bpi/dense_2/biasBpi/dense_2/bias/AdamBpi/dense_2/bias/Adam_1Bpi/dense_2/kernelBpi/dense_2/kernel/AdamBpi/dense_2/kernel/Adam_1Bv/dense/biasBv/dense/bias/AdamBv/dense/bias/Adam_1Bv/dense/kernelBv/dense/kernel/AdamBv/dense/kernel/Adam_1Bv/dense_1/biasBv/dense_1/bias/AdamBv/dense_1/bias/Adam_1Bv/dense_1/kernelBv/dense_1/kernel/AdamBv/dense_1/kernel/Adam_1Bv/dense_2/biasBv/dense_2/bias/AdamBv/dense_2/bias/Adam_1Bv/dense_2/kernelBv/dense_2/kernel/AdamBv/dense_2/kernel/Adam_1*
dtype0*
_output_shapes
:(
╣
"save_11/RestoreV2/shape_and_slicesConst*
dtype0*
_output_shapes
:(*c
valueZBX(B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 
Р
save_11/RestoreV2	RestoreV2save_11/Constsave_11/RestoreV2/tensor_names"save_11/RestoreV2/shape_and_slices*Х
_output_shapesБ
а::::::::::::::::::::::::::::::::::::::::*6
dtypes,
*2(
ц
save_11/AssignAssignbeta1_powersave_11/RestoreV2*
_output_shapes
: *
use_locking(*
T0* 
_class
loc:@pi/dense/bias*
validate_shape(
Е
save_11/Assign_1Assignbeta1_power_1save_11/RestoreV2:1*
use_locking(*
T0*
_class
loc:@v/dense/bias*
validate_shape(*
_output_shapes
: 
е
save_11/Assign_2Assignbeta2_powersave_11/RestoreV2:2*
_output_shapes
: *
use_locking(*
T0* 
_class
loc:@pi/dense/bias*
validate_shape(
Е
save_11/Assign_3Assignbeta2_power_1save_11/RestoreV2:3*
T0*
_class
loc:@v/dense/bias*
validate_shape(*
_output_shapes
: *
use_locking(
«
save_11/Assign_4Assignpi/dense/biassave_11/RestoreV2:4*
use_locking(*
T0* 
_class
loc:@pi/dense/bias*
validate_shape(*
_output_shapes
:@
│
save_11/Assign_5Assignpi/dense/bias/Adamsave_11/RestoreV2:5*
use_locking(*
T0* 
_class
loc:@pi/dense/bias*
validate_shape(*
_output_shapes
:@
х
save_11/Assign_6Assignpi/dense/bias/Adam_1save_11/RestoreV2:6*
validate_shape(*
_output_shapes
:@*
use_locking(*
T0* 
_class
loc:@pi/dense/bias
и
save_11/Assign_7Assignpi/dense/kernelsave_11/RestoreV2:7*
use_locking(*
T0*"
_class
loc:@pi/dense/kernel*
validate_shape(*
_output_shapes
:	Є@
╝
save_11/Assign_8Assignpi/dense/kernel/Adamsave_11/RestoreV2:8*
_output_shapes
:	Є@*
use_locking(*
T0*"
_class
loc:@pi/dense/kernel*
validate_shape(
Й
save_11/Assign_9Assignpi/dense/kernel/Adam_1save_11/RestoreV2:9*
_output_shapes
:	Є@*
use_locking(*
T0*"
_class
loc:@pi/dense/kernel*
validate_shape(
┤
save_11/Assign_10Assignpi/dense_1/biassave_11/RestoreV2:10*
use_locking(*
T0*"
_class
loc:@pi/dense_1/bias*
validate_shape(*
_output_shapes
:@
╣
save_11/Assign_11Assignpi/dense_1/bias/Adamsave_11/RestoreV2:11*
use_locking(*
T0*"
_class
loc:@pi/dense_1/bias*
validate_shape(*
_output_shapes
:@
╗
save_11/Assign_12Assignpi/dense_1/bias/Adam_1save_11/RestoreV2:12*
T0*"
_class
loc:@pi/dense_1/bias*
validate_shape(*
_output_shapes
:@*
use_locking(
╝
save_11/Assign_13Assignpi/dense_1/kernelsave_11/RestoreV2:13*
_output_shapes

:@@*
use_locking(*
T0*$
_class
loc:@pi/dense_1/kernel*
validate_shape(
┴
save_11/Assign_14Assignpi/dense_1/kernel/Adamsave_11/RestoreV2:14*
use_locking(*
T0*$
_class
loc:@pi/dense_1/kernel*
validate_shape(*
_output_shapes

:@@
├
save_11/Assign_15Assignpi/dense_1/kernel/Adam_1save_11/RestoreV2:15*
_output_shapes

:@@*
use_locking(*
T0*$
_class
loc:@pi/dense_1/kernel*
validate_shape(
┤
save_11/Assign_16Assignpi/dense_2/biassave_11/RestoreV2:16*
use_locking(*
T0*"
_class
loc:@pi/dense_2/bias*
validate_shape(*
_output_shapes
:
╣
save_11/Assign_17Assignpi/dense_2/bias/Adamsave_11/RestoreV2:17*
T0*"
_class
loc:@pi/dense_2/bias*
validate_shape(*
_output_shapes
:*
use_locking(
╗
save_11/Assign_18Assignpi/dense_2/bias/Adam_1save_11/RestoreV2:18*
T0*"
_class
loc:@pi/dense_2/bias*
validate_shape(*
_output_shapes
:*
use_locking(
╝
save_11/Assign_19Assignpi/dense_2/kernelsave_11/RestoreV2:19*$
_class
loc:@pi/dense_2/kernel*
validate_shape(*
_output_shapes

:@*
use_locking(*
T0
┴
save_11/Assign_20Assignpi/dense_2/kernel/Adamsave_11/RestoreV2:20*
use_locking(*
T0*$
_class
loc:@pi/dense_2/kernel*
validate_shape(*
_output_shapes

:@
├
save_11/Assign_21Assignpi/dense_2/kernel/Adam_1save_11/RestoreV2:21*
_output_shapes

:@*
use_locking(*
T0*$
_class
loc:@pi/dense_2/kernel*
validate_shape(
«
save_11/Assign_22Assignv/dense/biassave_11/RestoreV2:22*
T0*
_class
loc:@v/dense/bias*
validate_shape(*
_output_shapes
:@*
use_locking(
│
save_11/Assign_23Assignv/dense/bias/Adamsave_11/RestoreV2:23*
use_locking(*
T0*
_class
loc:@v/dense/bias*
validate_shape(*
_output_shapes
:@
х
save_11/Assign_24Assignv/dense/bias/Adam_1save_11/RestoreV2:24*
validate_shape(*
_output_shapes
:@*
use_locking(*
T0*
_class
loc:@v/dense/bias
и
save_11/Assign_25Assignv/dense/kernelsave_11/RestoreV2:25*
T0*!
_class
loc:@v/dense/kernel*
validate_shape(*
_output_shapes
:	Є@*
use_locking(
╝
save_11/Assign_26Assignv/dense/kernel/Adamsave_11/RestoreV2:26*
use_locking(*
T0*!
_class
loc:@v/dense/kernel*
validate_shape(*
_output_shapes
:	Є@
Й
save_11/Assign_27Assignv/dense/kernel/Adam_1save_11/RestoreV2:27*
T0*!
_class
loc:@v/dense/kernel*
validate_shape(*
_output_shapes
:	Є@*
use_locking(
▓
save_11/Assign_28Assignv/dense_1/biassave_11/RestoreV2:28*
use_locking(*
T0*!
_class
loc:@v/dense_1/bias*
validate_shape(*
_output_shapes
:@
и
save_11/Assign_29Assignv/dense_1/bias/Adamsave_11/RestoreV2:29*
validate_shape(*
_output_shapes
:@*
use_locking(*
T0*!
_class
loc:@v/dense_1/bias
╣
save_11/Assign_30Assignv/dense_1/bias/Adam_1save_11/RestoreV2:30*!
_class
loc:@v/dense_1/bias*
validate_shape(*
_output_shapes
:@*
use_locking(*
T0
║
save_11/Assign_31Assignv/dense_1/kernelsave_11/RestoreV2:31*
T0*#
_class
loc:@v/dense_1/kernel*
validate_shape(*
_output_shapes

:@@*
use_locking(
┐
save_11/Assign_32Assignv/dense_1/kernel/Adamsave_11/RestoreV2:32*
validate_shape(*
_output_shapes

:@@*
use_locking(*
T0*#
_class
loc:@v/dense_1/kernel
┴
save_11/Assign_33Assignv/dense_1/kernel/Adam_1save_11/RestoreV2:33*
use_locking(*
T0*#
_class
loc:@v/dense_1/kernel*
validate_shape(*
_output_shapes

:@@
▓
save_11/Assign_34Assignv/dense_2/biassave_11/RestoreV2:34*
validate_shape(*
_output_shapes
:*
use_locking(*
T0*!
_class
loc:@v/dense_2/bias
и
save_11/Assign_35Assignv/dense_2/bias/Adamsave_11/RestoreV2:35*
_output_shapes
:*
use_locking(*
T0*!
_class
loc:@v/dense_2/bias*
validate_shape(
╣
save_11/Assign_36Assignv/dense_2/bias/Adam_1save_11/RestoreV2:36*
T0*!
_class
loc:@v/dense_2/bias*
validate_shape(*
_output_shapes
:*
use_locking(
║
save_11/Assign_37Assignv/dense_2/kernelsave_11/RestoreV2:37*
use_locking(*
T0*#
_class
loc:@v/dense_2/kernel*
validate_shape(*
_output_shapes

:@
┐
save_11/Assign_38Assignv/dense_2/kernel/Adamsave_11/RestoreV2:38*
_output_shapes

:@*
use_locking(*
T0*#
_class
loc:@v/dense_2/kernel*
validate_shape(
┴
save_11/Assign_39Assignv/dense_2/kernel/Adam_1save_11/RestoreV2:39*
use_locking(*
T0*#
_class
loc:@v/dense_2/kernel*
validate_shape(*
_output_shapes

:@
▒
save_11/restore_shardNoOp^save_11/Assign^save_11/Assign_1^save_11/Assign_10^save_11/Assign_11^save_11/Assign_12^save_11/Assign_13^save_11/Assign_14^save_11/Assign_15^save_11/Assign_16^save_11/Assign_17^save_11/Assign_18^save_11/Assign_19^save_11/Assign_2^save_11/Assign_20^save_11/Assign_21^save_11/Assign_22^save_11/Assign_23^save_11/Assign_24^save_11/Assign_25^save_11/Assign_26^save_11/Assign_27^save_11/Assign_28^save_11/Assign_29^save_11/Assign_3^save_11/Assign_30^save_11/Assign_31^save_11/Assign_32^save_11/Assign_33^save_11/Assign_34^save_11/Assign_35^save_11/Assign_36^save_11/Assign_37^save_11/Assign_38^save_11/Assign_39^save_11/Assign_4^save_11/Assign_5^save_11/Assign_6^save_11/Assign_7^save_11/Assign_8^save_11/Assign_9
3
save_11/restore_allNoOp^save_11/restore_shard
\
save_12/filename/inputConst*
valueB Bmodel*
dtype0*
_output_shapes
: 
t
save_12/filenamePlaceholderWithDefaultsave_12/filename/input*
_output_shapes
: *
shape: *
dtype0
k
save_12/ConstPlaceholderWithDefaultsave_12/filename*
shape: *
dtype0*
_output_shapes
: 
Є
save_12/StringJoin/inputs_1Const*<
value3B1 B+_temp_82b9b40044be4a80afedce233a86eff1/part*
dtype0*
_output_shapes
: 
~
save_12/StringJoin
StringJoinsave_12/Constsave_12/StringJoin/inputs_1*
N*
_output_shapes
: *
	separator 
T
save_12/num_shardsConst*
value	B :*
dtype0*
_output_shapes
: 
_
save_12/ShardedFilename/shardConst*
value	B : *
dtype0*
_output_shapes
: 
Ѕ
save_12/ShardedFilenameShardedFilenamesave_12/StringJoinsave_12/ShardedFilename/shardsave_12/num_shards*
_output_shapes
: 
І
save_12/SaveV2/tensor_namesConst*╗
value▒B«(Bbeta1_powerBbeta1_power_1Bbeta2_powerBbeta2_power_1Bpi/dense/biasBpi/dense/bias/AdamBpi/dense/bias/Adam_1Bpi/dense/kernelBpi/dense/kernel/AdamBpi/dense/kernel/Adam_1Bpi/dense_1/biasBpi/dense_1/bias/AdamBpi/dense_1/bias/Adam_1Bpi/dense_1/kernelBpi/dense_1/kernel/AdamBpi/dense_1/kernel/Adam_1Bpi/dense_2/biasBpi/dense_2/bias/AdamBpi/dense_2/bias/Adam_1Bpi/dense_2/kernelBpi/dense_2/kernel/AdamBpi/dense_2/kernel/Adam_1Bv/dense/biasBv/dense/bias/AdamBv/dense/bias/Adam_1Bv/dense/kernelBv/dense/kernel/AdamBv/dense/kernel/Adam_1Bv/dense_1/biasBv/dense_1/bias/AdamBv/dense_1/bias/Adam_1Bv/dense_1/kernelBv/dense_1/kernel/AdamBv/dense_1/kernel/Adam_1Bv/dense_2/biasBv/dense_2/bias/AdamBv/dense_2/bias/Adam_1Bv/dense_2/kernelBv/dense_2/kernel/AdamBv/dense_2/kernel/Adam_1*
dtype0*
_output_shapes
:(
Х
save_12/SaveV2/shape_and_slicesConst*c
valueZBX(B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B *
dtype0*
_output_shapes
:(
═
save_12/SaveV2SaveV2save_12/ShardedFilenamesave_12/SaveV2/tensor_namessave_12/SaveV2/shape_and_slicesbeta1_powerbeta1_power_1beta2_powerbeta2_power_1pi/dense/biaspi/dense/bias/Adampi/dense/bias/Adam_1pi/dense/kernelpi/dense/kernel/Adampi/dense/kernel/Adam_1pi/dense_1/biaspi/dense_1/bias/Adampi/dense_1/bias/Adam_1pi/dense_1/kernelpi/dense_1/kernel/Adampi/dense_1/kernel/Adam_1pi/dense_2/biaspi/dense_2/bias/Adampi/dense_2/bias/Adam_1pi/dense_2/kernelpi/dense_2/kernel/Adampi/dense_2/kernel/Adam_1v/dense/biasv/dense/bias/Adamv/dense/bias/Adam_1v/dense/kernelv/dense/kernel/Adamv/dense/kernel/Adam_1v/dense_1/biasv/dense_1/bias/Adamv/dense_1/bias/Adam_1v/dense_1/kernelv/dense_1/kernel/Adamv/dense_1/kernel/Adam_1v/dense_2/biasv/dense_2/bias/Adamv/dense_2/bias/Adam_1v/dense_2/kernelv/dense_2/kernel/Adamv/dense_2/kernel/Adam_1*6
dtypes,
*2(
Ю
save_12/control_dependencyIdentitysave_12/ShardedFilename^save_12/SaveV2*
_output_shapes
: *
T0**
_class 
loc:@save_12/ShardedFilename
д
.save_12/MergeV2Checkpoints/checkpoint_prefixesPacksave_12/ShardedFilename^save_12/control_dependency*
T0*

axis *
N*
_output_shapes
:
є
save_12/MergeV2CheckpointsMergeV2Checkpoints.save_12/MergeV2Checkpoints/checkpoint_prefixessave_12/Const*
delete_old_dirs(
є
save_12/IdentityIdentitysave_12/Const^save_12/MergeV2Checkpoints^save_12/control_dependency*
_output_shapes
: *
T0
ј
save_12/RestoreV2/tensor_namesConst*╗
value▒B«(Bbeta1_powerBbeta1_power_1Bbeta2_powerBbeta2_power_1Bpi/dense/biasBpi/dense/bias/AdamBpi/dense/bias/Adam_1Bpi/dense/kernelBpi/dense/kernel/AdamBpi/dense/kernel/Adam_1Bpi/dense_1/biasBpi/dense_1/bias/AdamBpi/dense_1/bias/Adam_1Bpi/dense_1/kernelBpi/dense_1/kernel/AdamBpi/dense_1/kernel/Adam_1Bpi/dense_2/biasBpi/dense_2/bias/AdamBpi/dense_2/bias/Adam_1Bpi/dense_2/kernelBpi/dense_2/kernel/AdamBpi/dense_2/kernel/Adam_1Bv/dense/biasBv/dense/bias/AdamBv/dense/bias/Adam_1Bv/dense/kernelBv/dense/kernel/AdamBv/dense/kernel/Adam_1Bv/dense_1/biasBv/dense_1/bias/AdamBv/dense_1/bias/Adam_1Bv/dense_1/kernelBv/dense_1/kernel/AdamBv/dense_1/kernel/Adam_1Bv/dense_2/biasBv/dense_2/bias/AdamBv/dense_2/bias/Adam_1Bv/dense_2/kernelBv/dense_2/kernel/AdamBv/dense_2/kernel/Adam_1*
dtype0*
_output_shapes
:(
╣
"save_12/RestoreV2/shape_and_slicesConst*c
valueZBX(B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B *
dtype0*
_output_shapes
:(
Р
save_12/RestoreV2	RestoreV2save_12/Constsave_12/RestoreV2/tensor_names"save_12/RestoreV2/shape_and_slices*Х
_output_shapesБ
а::::::::::::::::::::::::::::::::::::::::*6
dtypes,
*2(
ц
save_12/AssignAssignbeta1_powersave_12/RestoreV2*
use_locking(*
T0* 
_class
loc:@pi/dense/bias*
validate_shape(*
_output_shapes
: 
Е
save_12/Assign_1Assignbeta1_power_1save_12/RestoreV2:1*
T0*
_class
loc:@v/dense/bias*
validate_shape(*
_output_shapes
: *
use_locking(
е
save_12/Assign_2Assignbeta2_powersave_12/RestoreV2:2*
use_locking(*
T0* 
_class
loc:@pi/dense/bias*
validate_shape(*
_output_shapes
: 
Е
save_12/Assign_3Assignbeta2_power_1save_12/RestoreV2:3*
_class
loc:@v/dense/bias*
validate_shape(*
_output_shapes
: *
use_locking(*
T0
«
save_12/Assign_4Assignpi/dense/biassave_12/RestoreV2:4*
T0* 
_class
loc:@pi/dense/bias*
validate_shape(*
_output_shapes
:@*
use_locking(
│
save_12/Assign_5Assignpi/dense/bias/Adamsave_12/RestoreV2:5*
use_locking(*
T0* 
_class
loc:@pi/dense/bias*
validate_shape(*
_output_shapes
:@
х
save_12/Assign_6Assignpi/dense/bias/Adam_1save_12/RestoreV2:6*
validate_shape(*
_output_shapes
:@*
use_locking(*
T0* 
_class
loc:@pi/dense/bias
и
save_12/Assign_7Assignpi/dense/kernelsave_12/RestoreV2:7*
use_locking(*
T0*"
_class
loc:@pi/dense/kernel*
validate_shape(*
_output_shapes
:	Є@
╝
save_12/Assign_8Assignpi/dense/kernel/Adamsave_12/RestoreV2:8*
validate_shape(*
_output_shapes
:	Є@*
use_locking(*
T0*"
_class
loc:@pi/dense/kernel
Й
save_12/Assign_9Assignpi/dense/kernel/Adam_1save_12/RestoreV2:9*
use_locking(*
T0*"
_class
loc:@pi/dense/kernel*
validate_shape(*
_output_shapes
:	Є@
┤
save_12/Assign_10Assignpi/dense_1/biassave_12/RestoreV2:10*
use_locking(*
T0*"
_class
loc:@pi/dense_1/bias*
validate_shape(*
_output_shapes
:@
╣
save_12/Assign_11Assignpi/dense_1/bias/Adamsave_12/RestoreV2:11*
validate_shape(*
_output_shapes
:@*
use_locking(*
T0*"
_class
loc:@pi/dense_1/bias
╗
save_12/Assign_12Assignpi/dense_1/bias/Adam_1save_12/RestoreV2:12*
use_locking(*
T0*"
_class
loc:@pi/dense_1/bias*
validate_shape(*
_output_shapes
:@
╝
save_12/Assign_13Assignpi/dense_1/kernelsave_12/RestoreV2:13*
use_locking(*
T0*$
_class
loc:@pi/dense_1/kernel*
validate_shape(*
_output_shapes

:@@
┴
save_12/Assign_14Assignpi/dense_1/kernel/Adamsave_12/RestoreV2:14*
use_locking(*
T0*$
_class
loc:@pi/dense_1/kernel*
validate_shape(*
_output_shapes

:@@
├
save_12/Assign_15Assignpi/dense_1/kernel/Adam_1save_12/RestoreV2:15*
T0*$
_class
loc:@pi/dense_1/kernel*
validate_shape(*
_output_shapes

:@@*
use_locking(
┤
save_12/Assign_16Assignpi/dense_2/biassave_12/RestoreV2:16*
T0*"
_class
loc:@pi/dense_2/bias*
validate_shape(*
_output_shapes
:*
use_locking(
╣
save_12/Assign_17Assignpi/dense_2/bias/Adamsave_12/RestoreV2:17*
use_locking(*
T0*"
_class
loc:@pi/dense_2/bias*
validate_shape(*
_output_shapes
:
╗
save_12/Assign_18Assignpi/dense_2/bias/Adam_1save_12/RestoreV2:18*
use_locking(*
T0*"
_class
loc:@pi/dense_2/bias*
validate_shape(*
_output_shapes
:
╝
save_12/Assign_19Assignpi/dense_2/kernelsave_12/RestoreV2:19*
T0*$
_class
loc:@pi/dense_2/kernel*
validate_shape(*
_output_shapes

:@*
use_locking(
┴
save_12/Assign_20Assignpi/dense_2/kernel/Adamsave_12/RestoreV2:20*
use_locking(*
T0*$
_class
loc:@pi/dense_2/kernel*
validate_shape(*
_output_shapes

:@
├
save_12/Assign_21Assignpi/dense_2/kernel/Adam_1save_12/RestoreV2:21*
use_locking(*
T0*$
_class
loc:@pi/dense_2/kernel*
validate_shape(*
_output_shapes

:@
«
save_12/Assign_22Assignv/dense/biassave_12/RestoreV2:22*
use_locking(*
T0*
_class
loc:@v/dense/bias*
validate_shape(*
_output_shapes
:@
│
save_12/Assign_23Assignv/dense/bias/Adamsave_12/RestoreV2:23*
use_locking(*
T0*
_class
loc:@v/dense/bias*
validate_shape(*
_output_shapes
:@
х
save_12/Assign_24Assignv/dense/bias/Adam_1save_12/RestoreV2:24*
T0*
_class
loc:@v/dense/bias*
validate_shape(*
_output_shapes
:@*
use_locking(
и
save_12/Assign_25Assignv/dense/kernelsave_12/RestoreV2:25*
use_locking(*
T0*!
_class
loc:@v/dense/kernel*
validate_shape(*
_output_shapes
:	Є@
╝
save_12/Assign_26Assignv/dense/kernel/Adamsave_12/RestoreV2:26*
use_locking(*
T0*!
_class
loc:@v/dense/kernel*
validate_shape(*
_output_shapes
:	Є@
Й
save_12/Assign_27Assignv/dense/kernel/Adam_1save_12/RestoreV2:27*
T0*!
_class
loc:@v/dense/kernel*
validate_shape(*
_output_shapes
:	Є@*
use_locking(
▓
save_12/Assign_28Assignv/dense_1/biassave_12/RestoreV2:28*
T0*!
_class
loc:@v/dense_1/bias*
validate_shape(*
_output_shapes
:@*
use_locking(
и
save_12/Assign_29Assignv/dense_1/bias/Adamsave_12/RestoreV2:29*
_output_shapes
:@*
use_locking(*
T0*!
_class
loc:@v/dense_1/bias*
validate_shape(
╣
save_12/Assign_30Assignv/dense_1/bias/Adam_1save_12/RestoreV2:30*
T0*!
_class
loc:@v/dense_1/bias*
validate_shape(*
_output_shapes
:@*
use_locking(
║
save_12/Assign_31Assignv/dense_1/kernelsave_12/RestoreV2:31*
use_locking(*
T0*#
_class
loc:@v/dense_1/kernel*
validate_shape(*
_output_shapes

:@@
┐
save_12/Assign_32Assignv/dense_1/kernel/Adamsave_12/RestoreV2:32*
T0*#
_class
loc:@v/dense_1/kernel*
validate_shape(*
_output_shapes

:@@*
use_locking(
┴
save_12/Assign_33Assignv/dense_1/kernel/Adam_1save_12/RestoreV2:33*
use_locking(*
T0*#
_class
loc:@v/dense_1/kernel*
validate_shape(*
_output_shapes

:@@
▓
save_12/Assign_34Assignv/dense_2/biassave_12/RestoreV2:34*
T0*!
_class
loc:@v/dense_2/bias*
validate_shape(*
_output_shapes
:*
use_locking(
и
save_12/Assign_35Assignv/dense_2/bias/Adamsave_12/RestoreV2:35*!
_class
loc:@v/dense_2/bias*
validate_shape(*
_output_shapes
:*
use_locking(*
T0
╣
save_12/Assign_36Assignv/dense_2/bias/Adam_1save_12/RestoreV2:36*
validate_shape(*
_output_shapes
:*
use_locking(*
T0*!
_class
loc:@v/dense_2/bias
║
save_12/Assign_37Assignv/dense_2/kernelsave_12/RestoreV2:37*#
_class
loc:@v/dense_2/kernel*
validate_shape(*
_output_shapes

:@*
use_locking(*
T0
┐
save_12/Assign_38Assignv/dense_2/kernel/Adamsave_12/RestoreV2:38*
T0*#
_class
loc:@v/dense_2/kernel*
validate_shape(*
_output_shapes

:@*
use_locking(
┴
save_12/Assign_39Assignv/dense_2/kernel/Adam_1save_12/RestoreV2:39*
use_locking(*
T0*#
_class
loc:@v/dense_2/kernel*
validate_shape(*
_output_shapes

:@
▒
save_12/restore_shardNoOp^save_12/Assign^save_12/Assign_1^save_12/Assign_10^save_12/Assign_11^save_12/Assign_12^save_12/Assign_13^save_12/Assign_14^save_12/Assign_15^save_12/Assign_16^save_12/Assign_17^save_12/Assign_18^save_12/Assign_19^save_12/Assign_2^save_12/Assign_20^save_12/Assign_21^save_12/Assign_22^save_12/Assign_23^save_12/Assign_24^save_12/Assign_25^save_12/Assign_26^save_12/Assign_27^save_12/Assign_28^save_12/Assign_29^save_12/Assign_3^save_12/Assign_30^save_12/Assign_31^save_12/Assign_32^save_12/Assign_33^save_12/Assign_34^save_12/Assign_35^save_12/Assign_36^save_12/Assign_37^save_12/Assign_38^save_12/Assign_39^save_12/Assign_4^save_12/Assign_5^save_12/Assign_6^save_12/Assign_7^save_12/Assign_8^save_12/Assign_9
3
save_12/restore_allNoOp^save_12/restore_shard
\
save_13/filename/inputConst*
valueB Bmodel*
dtype0*
_output_shapes
: 
t
save_13/filenamePlaceholderWithDefaultsave_13/filename/input*
_output_shapes
: *
shape: *
dtype0
k
save_13/ConstPlaceholderWithDefaultsave_13/filename*
_output_shapes
: *
shape: *
dtype0
Є
save_13/StringJoin/inputs_1Const*<
value3B1 B+_temp_08c23639dd32484d9477daea047611a4/part*
dtype0*
_output_shapes
: 
~
save_13/StringJoin
StringJoinsave_13/Constsave_13/StringJoin/inputs_1*
	separator *
N*
_output_shapes
: 
T
save_13/num_shardsConst*
value	B :*
dtype0*
_output_shapes
: 
_
save_13/ShardedFilename/shardConst*
value	B : *
dtype0*
_output_shapes
: 
Ѕ
save_13/ShardedFilenameShardedFilenamesave_13/StringJoinsave_13/ShardedFilename/shardsave_13/num_shards*
_output_shapes
: 
І
save_13/SaveV2/tensor_namesConst*╗
value▒B«(Bbeta1_powerBbeta1_power_1Bbeta2_powerBbeta2_power_1Bpi/dense/biasBpi/dense/bias/AdamBpi/dense/bias/Adam_1Bpi/dense/kernelBpi/dense/kernel/AdamBpi/dense/kernel/Adam_1Bpi/dense_1/biasBpi/dense_1/bias/AdamBpi/dense_1/bias/Adam_1Bpi/dense_1/kernelBpi/dense_1/kernel/AdamBpi/dense_1/kernel/Adam_1Bpi/dense_2/biasBpi/dense_2/bias/AdamBpi/dense_2/bias/Adam_1Bpi/dense_2/kernelBpi/dense_2/kernel/AdamBpi/dense_2/kernel/Adam_1Bv/dense/biasBv/dense/bias/AdamBv/dense/bias/Adam_1Bv/dense/kernelBv/dense/kernel/AdamBv/dense/kernel/Adam_1Bv/dense_1/biasBv/dense_1/bias/AdamBv/dense_1/bias/Adam_1Bv/dense_1/kernelBv/dense_1/kernel/AdamBv/dense_1/kernel/Adam_1Bv/dense_2/biasBv/dense_2/bias/AdamBv/dense_2/bias/Adam_1Bv/dense_2/kernelBv/dense_2/kernel/AdamBv/dense_2/kernel/Adam_1*
dtype0*
_output_shapes
:(
Х
save_13/SaveV2/shape_and_slicesConst*c
valueZBX(B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B *
dtype0*
_output_shapes
:(
═
save_13/SaveV2SaveV2save_13/ShardedFilenamesave_13/SaveV2/tensor_namessave_13/SaveV2/shape_and_slicesbeta1_powerbeta1_power_1beta2_powerbeta2_power_1pi/dense/biaspi/dense/bias/Adampi/dense/bias/Adam_1pi/dense/kernelpi/dense/kernel/Adampi/dense/kernel/Adam_1pi/dense_1/biaspi/dense_1/bias/Adampi/dense_1/bias/Adam_1pi/dense_1/kernelpi/dense_1/kernel/Adampi/dense_1/kernel/Adam_1pi/dense_2/biaspi/dense_2/bias/Adampi/dense_2/bias/Adam_1pi/dense_2/kernelpi/dense_2/kernel/Adampi/dense_2/kernel/Adam_1v/dense/biasv/dense/bias/Adamv/dense/bias/Adam_1v/dense/kernelv/dense/kernel/Adamv/dense/kernel/Adam_1v/dense_1/biasv/dense_1/bias/Adamv/dense_1/bias/Adam_1v/dense_1/kernelv/dense_1/kernel/Adamv/dense_1/kernel/Adam_1v/dense_2/biasv/dense_2/bias/Adamv/dense_2/bias/Adam_1v/dense_2/kernelv/dense_2/kernel/Adamv/dense_2/kernel/Adam_1*6
dtypes,
*2(
Ю
save_13/control_dependencyIdentitysave_13/ShardedFilename^save_13/SaveV2*
T0**
_class 
loc:@save_13/ShardedFilename*
_output_shapes
: 
д
.save_13/MergeV2Checkpoints/checkpoint_prefixesPacksave_13/ShardedFilename^save_13/control_dependency*
T0*

axis *
N*
_output_shapes
:
є
save_13/MergeV2CheckpointsMergeV2Checkpoints.save_13/MergeV2Checkpoints/checkpoint_prefixessave_13/Const*
delete_old_dirs(
є
save_13/IdentityIdentitysave_13/Const^save_13/MergeV2Checkpoints^save_13/control_dependency*
T0*
_output_shapes
: 
ј
save_13/RestoreV2/tensor_namesConst*╗
value▒B«(Bbeta1_powerBbeta1_power_1Bbeta2_powerBbeta2_power_1Bpi/dense/biasBpi/dense/bias/AdamBpi/dense/bias/Adam_1Bpi/dense/kernelBpi/dense/kernel/AdamBpi/dense/kernel/Adam_1Bpi/dense_1/biasBpi/dense_1/bias/AdamBpi/dense_1/bias/Adam_1Bpi/dense_1/kernelBpi/dense_1/kernel/AdamBpi/dense_1/kernel/Adam_1Bpi/dense_2/biasBpi/dense_2/bias/AdamBpi/dense_2/bias/Adam_1Bpi/dense_2/kernelBpi/dense_2/kernel/AdamBpi/dense_2/kernel/Adam_1Bv/dense/biasBv/dense/bias/AdamBv/dense/bias/Adam_1Bv/dense/kernelBv/dense/kernel/AdamBv/dense/kernel/Adam_1Bv/dense_1/biasBv/dense_1/bias/AdamBv/dense_1/bias/Adam_1Bv/dense_1/kernelBv/dense_1/kernel/AdamBv/dense_1/kernel/Adam_1Bv/dense_2/biasBv/dense_2/bias/AdamBv/dense_2/bias/Adam_1Bv/dense_2/kernelBv/dense_2/kernel/AdamBv/dense_2/kernel/Adam_1*
dtype0*
_output_shapes
:(
╣
"save_13/RestoreV2/shape_and_slicesConst*c
valueZBX(B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B *
dtype0*
_output_shapes
:(
Р
save_13/RestoreV2	RestoreV2save_13/Constsave_13/RestoreV2/tensor_names"save_13/RestoreV2/shape_and_slices*Х
_output_shapesБ
а::::::::::::::::::::::::::::::::::::::::*6
dtypes,
*2(
ц
save_13/AssignAssignbeta1_powersave_13/RestoreV2*
validate_shape(*
_output_shapes
: *
use_locking(*
T0* 
_class
loc:@pi/dense/bias
Е
save_13/Assign_1Assignbeta1_power_1save_13/RestoreV2:1*
_output_shapes
: *
use_locking(*
T0*
_class
loc:@v/dense/bias*
validate_shape(
е
save_13/Assign_2Assignbeta2_powersave_13/RestoreV2:2*
use_locking(*
T0* 
_class
loc:@pi/dense/bias*
validate_shape(*
_output_shapes
: 
Е
save_13/Assign_3Assignbeta2_power_1save_13/RestoreV2:3*
validate_shape(*
_output_shapes
: *
use_locking(*
T0*
_class
loc:@v/dense/bias
«
save_13/Assign_4Assignpi/dense/biassave_13/RestoreV2:4*
use_locking(*
T0* 
_class
loc:@pi/dense/bias*
validate_shape(*
_output_shapes
:@
│
save_13/Assign_5Assignpi/dense/bias/Adamsave_13/RestoreV2:5*
use_locking(*
T0* 
_class
loc:@pi/dense/bias*
validate_shape(*
_output_shapes
:@
х
save_13/Assign_6Assignpi/dense/bias/Adam_1save_13/RestoreV2:6*
validate_shape(*
_output_shapes
:@*
use_locking(*
T0* 
_class
loc:@pi/dense/bias
и
save_13/Assign_7Assignpi/dense/kernelsave_13/RestoreV2:7*
_output_shapes
:	Є@*
use_locking(*
T0*"
_class
loc:@pi/dense/kernel*
validate_shape(
╝
save_13/Assign_8Assignpi/dense/kernel/Adamsave_13/RestoreV2:8*
use_locking(*
T0*"
_class
loc:@pi/dense/kernel*
validate_shape(*
_output_shapes
:	Є@
Й
save_13/Assign_9Assignpi/dense/kernel/Adam_1save_13/RestoreV2:9*
use_locking(*
T0*"
_class
loc:@pi/dense/kernel*
validate_shape(*
_output_shapes
:	Є@
┤
save_13/Assign_10Assignpi/dense_1/biassave_13/RestoreV2:10*
use_locking(*
T0*"
_class
loc:@pi/dense_1/bias*
validate_shape(*
_output_shapes
:@
╣
save_13/Assign_11Assignpi/dense_1/bias/Adamsave_13/RestoreV2:11*
use_locking(*
T0*"
_class
loc:@pi/dense_1/bias*
validate_shape(*
_output_shapes
:@
╗
save_13/Assign_12Assignpi/dense_1/bias/Adam_1save_13/RestoreV2:12*
_output_shapes
:@*
use_locking(*
T0*"
_class
loc:@pi/dense_1/bias*
validate_shape(
╝
save_13/Assign_13Assignpi/dense_1/kernelsave_13/RestoreV2:13*
use_locking(*
T0*$
_class
loc:@pi/dense_1/kernel*
validate_shape(*
_output_shapes

:@@
┴
save_13/Assign_14Assignpi/dense_1/kernel/Adamsave_13/RestoreV2:14*
validate_shape(*
_output_shapes

:@@*
use_locking(*
T0*$
_class
loc:@pi/dense_1/kernel
├
save_13/Assign_15Assignpi/dense_1/kernel/Adam_1save_13/RestoreV2:15*
_output_shapes

:@@*
use_locking(*
T0*$
_class
loc:@pi/dense_1/kernel*
validate_shape(
┤
save_13/Assign_16Assignpi/dense_2/biassave_13/RestoreV2:16*
validate_shape(*
_output_shapes
:*
use_locking(*
T0*"
_class
loc:@pi/dense_2/bias
╣
save_13/Assign_17Assignpi/dense_2/bias/Adamsave_13/RestoreV2:17*
_output_shapes
:*
use_locking(*
T0*"
_class
loc:@pi/dense_2/bias*
validate_shape(
╗
save_13/Assign_18Assignpi/dense_2/bias/Adam_1save_13/RestoreV2:18*
T0*"
_class
loc:@pi/dense_2/bias*
validate_shape(*
_output_shapes
:*
use_locking(
╝
save_13/Assign_19Assignpi/dense_2/kernelsave_13/RestoreV2:19*$
_class
loc:@pi/dense_2/kernel*
validate_shape(*
_output_shapes

:@*
use_locking(*
T0
┴
save_13/Assign_20Assignpi/dense_2/kernel/Adamsave_13/RestoreV2:20*
use_locking(*
T0*$
_class
loc:@pi/dense_2/kernel*
validate_shape(*
_output_shapes

:@
├
save_13/Assign_21Assignpi/dense_2/kernel/Adam_1save_13/RestoreV2:21*
use_locking(*
T0*$
_class
loc:@pi/dense_2/kernel*
validate_shape(*
_output_shapes

:@
«
save_13/Assign_22Assignv/dense/biassave_13/RestoreV2:22*
_output_shapes
:@*
use_locking(*
T0*
_class
loc:@v/dense/bias*
validate_shape(
│
save_13/Assign_23Assignv/dense/bias/Adamsave_13/RestoreV2:23*
use_locking(*
T0*
_class
loc:@v/dense/bias*
validate_shape(*
_output_shapes
:@
х
save_13/Assign_24Assignv/dense/bias/Adam_1save_13/RestoreV2:24*
use_locking(*
T0*
_class
loc:@v/dense/bias*
validate_shape(*
_output_shapes
:@
и
save_13/Assign_25Assignv/dense/kernelsave_13/RestoreV2:25*
T0*!
_class
loc:@v/dense/kernel*
validate_shape(*
_output_shapes
:	Є@*
use_locking(
╝
save_13/Assign_26Assignv/dense/kernel/Adamsave_13/RestoreV2:26*
_output_shapes
:	Є@*
use_locking(*
T0*!
_class
loc:@v/dense/kernel*
validate_shape(
Й
save_13/Assign_27Assignv/dense/kernel/Adam_1save_13/RestoreV2:27*!
_class
loc:@v/dense/kernel*
validate_shape(*
_output_shapes
:	Є@*
use_locking(*
T0
▓
save_13/Assign_28Assignv/dense_1/biassave_13/RestoreV2:28*
use_locking(*
T0*!
_class
loc:@v/dense_1/bias*
validate_shape(*
_output_shapes
:@
и
save_13/Assign_29Assignv/dense_1/bias/Adamsave_13/RestoreV2:29*
use_locking(*
T0*!
_class
loc:@v/dense_1/bias*
validate_shape(*
_output_shapes
:@
╣
save_13/Assign_30Assignv/dense_1/bias/Adam_1save_13/RestoreV2:30*
use_locking(*
T0*!
_class
loc:@v/dense_1/bias*
validate_shape(*
_output_shapes
:@
║
save_13/Assign_31Assignv/dense_1/kernelsave_13/RestoreV2:31*
validate_shape(*
_output_shapes

:@@*
use_locking(*
T0*#
_class
loc:@v/dense_1/kernel
┐
save_13/Assign_32Assignv/dense_1/kernel/Adamsave_13/RestoreV2:32*
use_locking(*
T0*#
_class
loc:@v/dense_1/kernel*
validate_shape(*
_output_shapes

:@@
┴
save_13/Assign_33Assignv/dense_1/kernel/Adam_1save_13/RestoreV2:33*#
_class
loc:@v/dense_1/kernel*
validate_shape(*
_output_shapes

:@@*
use_locking(*
T0
▓
save_13/Assign_34Assignv/dense_2/biassave_13/RestoreV2:34*
validate_shape(*
_output_shapes
:*
use_locking(*
T0*!
_class
loc:@v/dense_2/bias
и
save_13/Assign_35Assignv/dense_2/bias/Adamsave_13/RestoreV2:35*
_output_shapes
:*
use_locking(*
T0*!
_class
loc:@v/dense_2/bias*
validate_shape(
╣
save_13/Assign_36Assignv/dense_2/bias/Adam_1save_13/RestoreV2:36*
T0*!
_class
loc:@v/dense_2/bias*
validate_shape(*
_output_shapes
:*
use_locking(
║
save_13/Assign_37Assignv/dense_2/kernelsave_13/RestoreV2:37*
T0*#
_class
loc:@v/dense_2/kernel*
validate_shape(*
_output_shapes

:@*
use_locking(
┐
save_13/Assign_38Assignv/dense_2/kernel/Adamsave_13/RestoreV2:38*
use_locking(*
T0*#
_class
loc:@v/dense_2/kernel*
validate_shape(*
_output_shapes

:@
┴
save_13/Assign_39Assignv/dense_2/kernel/Adam_1save_13/RestoreV2:39*
use_locking(*
T0*#
_class
loc:@v/dense_2/kernel*
validate_shape(*
_output_shapes

:@
▒
save_13/restore_shardNoOp^save_13/Assign^save_13/Assign_1^save_13/Assign_10^save_13/Assign_11^save_13/Assign_12^save_13/Assign_13^save_13/Assign_14^save_13/Assign_15^save_13/Assign_16^save_13/Assign_17^save_13/Assign_18^save_13/Assign_19^save_13/Assign_2^save_13/Assign_20^save_13/Assign_21^save_13/Assign_22^save_13/Assign_23^save_13/Assign_24^save_13/Assign_25^save_13/Assign_26^save_13/Assign_27^save_13/Assign_28^save_13/Assign_29^save_13/Assign_3^save_13/Assign_30^save_13/Assign_31^save_13/Assign_32^save_13/Assign_33^save_13/Assign_34^save_13/Assign_35^save_13/Assign_36^save_13/Assign_37^save_13/Assign_38^save_13/Assign_39^save_13/Assign_4^save_13/Assign_5^save_13/Assign_6^save_13/Assign_7^save_13/Assign_8^save_13/Assign_9
3
save_13/restore_allNoOp^save_13/restore_shard
\
save_14/filename/inputConst*
valueB Bmodel*
dtype0*
_output_shapes
: 
t
save_14/filenamePlaceholderWithDefaultsave_14/filename/input*
shape: *
dtype0*
_output_shapes
: 
k
save_14/ConstPlaceholderWithDefaultsave_14/filename*
shape: *
dtype0*
_output_shapes
: 
Є
save_14/StringJoin/inputs_1Const*<
value3B1 B+_temp_f16f945c6d9f48c79afbffdbb0256de4/part*
dtype0*
_output_shapes
: 
~
save_14/StringJoin
StringJoinsave_14/Constsave_14/StringJoin/inputs_1*
	separator *
N*
_output_shapes
: 
T
save_14/num_shardsConst*
value	B :*
dtype0*
_output_shapes
: 
_
save_14/ShardedFilename/shardConst*
_output_shapes
: *
value	B : *
dtype0
Ѕ
save_14/ShardedFilenameShardedFilenamesave_14/StringJoinsave_14/ShardedFilename/shardsave_14/num_shards*
_output_shapes
: 
І
save_14/SaveV2/tensor_namesConst*╗
value▒B«(Bbeta1_powerBbeta1_power_1Bbeta2_powerBbeta2_power_1Bpi/dense/biasBpi/dense/bias/AdamBpi/dense/bias/Adam_1Bpi/dense/kernelBpi/dense/kernel/AdamBpi/dense/kernel/Adam_1Bpi/dense_1/biasBpi/dense_1/bias/AdamBpi/dense_1/bias/Adam_1Bpi/dense_1/kernelBpi/dense_1/kernel/AdamBpi/dense_1/kernel/Adam_1Bpi/dense_2/biasBpi/dense_2/bias/AdamBpi/dense_2/bias/Adam_1Bpi/dense_2/kernelBpi/dense_2/kernel/AdamBpi/dense_2/kernel/Adam_1Bv/dense/biasBv/dense/bias/AdamBv/dense/bias/Adam_1Bv/dense/kernelBv/dense/kernel/AdamBv/dense/kernel/Adam_1Bv/dense_1/biasBv/dense_1/bias/AdamBv/dense_1/bias/Adam_1Bv/dense_1/kernelBv/dense_1/kernel/AdamBv/dense_1/kernel/Adam_1Bv/dense_2/biasBv/dense_2/bias/AdamBv/dense_2/bias/Adam_1Bv/dense_2/kernelBv/dense_2/kernel/AdamBv/dense_2/kernel/Adam_1*
dtype0*
_output_shapes
:(
Х
save_14/SaveV2/shape_and_slicesConst*
dtype0*
_output_shapes
:(*c
valueZBX(B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 
═
save_14/SaveV2SaveV2save_14/ShardedFilenamesave_14/SaveV2/tensor_namessave_14/SaveV2/shape_and_slicesbeta1_powerbeta1_power_1beta2_powerbeta2_power_1pi/dense/biaspi/dense/bias/Adampi/dense/bias/Adam_1pi/dense/kernelpi/dense/kernel/Adampi/dense/kernel/Adam_1pi/dense_1/biaspi/dense_1/bias/Adampi/dense_1/bias/Adam_1pi/dense_1/kernelpi/dense_1/kernel/Adampi/dense_1/kernel/Adam_1pi/dense_2/biaspi/dense_2/bias/Adampi/dense_2/bias/Adam_1pi/dense_2/kernelpi/dense_2/kernel/Adampi/dense_2/kernel/Adam_1v/dense/biasv/dense/bias/Adamv/dense/bias/Adam_1v/dense/kernelv/dense/kernel/Adamv/dense/kernel/Adam_1v/dense_1/biasv/dense_1/bias/Adamv/dense_1/bias/Adam_1v/dense_1/kernelv/dense_1/kernel/Adamv/dense_1/kernel/Adam_1v/dense_2/biasv/dense_2/bias/Adamv/dense_2/bias/Adam_1v/dense_2/kernelv/dense_2/kernel/Adamv/dense_2/kernel/Adam_1*6
dtypes,
*2(
Ю
save_14/control_dependencyIdentitysave_14/ShardedFilename^save_14/SaveV2*
T0**
_class 
loc:@save_14/ShardedFilename*
_output_shapes
: 
д
.save_14/MergeV2Checkpoints/checkpoint_prefixesPacksave_14/ShardedFilename^save_14/control_dependency*
T0*

axis *
N*
_output_shapes
:
є
save_14/MergeV2CheckpointsMergeV2Checkpoints.save_14/MergeV2Checkpoints/checkpoint_prefixessave_14/Const*
delete_old_dirs(
є
save_14/IdentityIdentitysave_14/Const^save_14/MergeV2Checkpoints^save_14/control_dependency*
T0*
_output_shapes
: 
ј
save_14/RestoreV2/tensor_namesConst*╗
value▒B«(Bbeta1_powerBbeta1_power_1Bbeta2_powerBbeta2_power_1Bpi/dense/biasBpi/dense/bias/AdamBpi/dense/bias/Adam_1Bpi/dense/kernelBpi/dense/kernel/AdamBpi/dense/kernel/Adam_1Bpi/dense_1/biasBpi/dense_1/bias/AdamBpi/dense_1/bias/Adam_1Bpi/dense_1/kernelBpi/dense_1/kernel/AdamBpi/dense_1/kernel/Adam_1Bpi/dense_2/biasBpi/dense_2/bias/AdamBpi/dense_2/bias/Adam_1Bpi/dense_2/kernelBpi/dense_2/kernel/AdamBpi/dense_2/kernel/Adam_1Bv/dense/biasBv/dense/bias/AdamBv/dense/bias/Adam_1Bv/dense/kernelBv/dense/kernel/AdamBv/dense/kernel/Adam_1Bv/dense_1/biasBv/dense_1/bias/AdamBv/dense_1/bias/Adam_1Bv/dense_1/kernelBv/dense_1/kernel/AdamBv/dense_1/kernel/Adam_1Bv/dense_2/biasBv/dense_2/bias/AdamBv/dense_2/bias/Adam_1Bv/dense_2/kernelBv/dense_2/kernel/AdamBv/dense_2/kernel/Adam_1*
dtype0*
_output_shapes
:(
╣
"save_14/RestoreV2/shape_and_slicesConst*
_output_shapes
:(*c
valueZBX(B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B *
dtype0
Р
save_14/RestoreV2	RestoreV2save_14/Constsave_14/RestoreV2/tensor_names"save_14/RestoreV2/shape_and_slices*Х
_output_shapesБ
а::::::::::::::::::::::::::::::::::::::::*6
dtypes,
*2(
ц
save_14/AssignAssignbeta1_powersave_14/RestoreV2*
T0* 
_class
loc:@pi/dense/bias*
validate_shape(*
_output_shapes
: *
use_locking(
Е
save_14/Assign_1Assignbeta1_power_1save_14/RestoreV2:1*
T0*
_class
loc:@v/dense/bias*
validate_shape(*
_output_shapes
: *
use_locking(
е
save_14/Assign_2Assignbeta2_powersave_14/RestoreV2:2*
use_locking(*
T0* 
_class
loc:@pi/dense/bias*
validate_shape(*
_output_shapes
: 
Е
save_14/Assign_3Assignbeta2_power_1save_14/RestoreV2:3*
validate_shape(*
_output_shapes
: *
use_locking(*
T0*
_class
loc:@v/dense/bias
«
save_14/Assign_4Assignpi/dense/biassave_14/RestoreV2:4*
_output_shapes
:@*
use_locking(*
T0* 
_class
loc:@pi/dense/bias*
validate_shape(
│
save_14/Assign_5Assignpi/dense/bias/Adamsave_14/RestoreV2:5*
use_locking(*
T0* 
_class
loc:@pi/dense/bias*
validate_shape(*
_output_shapes
:@
х
save_14/Assign_6Assignpi/dense/bias/Adam_1save_14/RestoreV2:6*
use_locking(*
T0* 
_class
loc:@pi/dense/bias*
validate_shape(*
_output_shapes
:@
и
save_14/Assign_7Assignpi/dense/kernelsave_14/RestoreV2:7*
use_locking(*
T0*"
_class
loc:@pi/dense/kernel*
validate_shape(*
_output_shapes
:	Є@
╝
save_14/Assign_8Assignpi/dense/kernel/Adamsave_14/RestoreV2:8*
_output_shapes
:	Є@*
use_locking(*
T0*"
_class
loc:@pi/dense/kernel*
validate_shape(
Й
save_14/Assign_9Assignpi/dense/kernel/Adam_1save_14/RestoreV2:9*"
_class
loc:@pi/dense/kernel*
validate_shape(*
_output_shapes
:	Є@*
use_locking(*
T0
┤
save_14/Assign_10Assignpi/dense_1/biassave_14/RestoreV2:10*
_output_shapes
:@*
use_locking(*
T0*"
_class
loc:@pi/dense_1/bias*
validate_shape(
╣
save_14/Assign_11Assignpi/dense_1/bias/Adamsave_14/RestoreV2:11*
use_locking(*
T0*"
_class
loc:@pi/dense_1/bias*
validate_shape(*
_output_shapes
:@
╗
save_14/Assign_12Assignpi/dense_1/bias/Adam_1save_14/RestoreV2:12*
use_locking(*
T0*"
_class
loc:@pi/dense_1/bias*
validate_shape(*
_output_shapes
:@
╝
save_14/Assign_13Assignpi/dense_1/kernelsave_14/RestoreV2:13*
use_locking(*
T0*$
_class
loc:@pi/dense_1/kernel*
validate_shape(*
_output_shapes

:@@
┴
save_14/Assign_14Assignpi/dense_1/kernel/Adamsave_14/RestoreV2:14*
T0*$
_class
loc:@pi/dense_1/kernel*
validate_shape(*
_output_shapes

:@@*
use_locking(
├
save_14/Assign_15Assignpi/dense_1/kernel/Adam_1save_14/RestoreV2:15*
validate_shape(*
_output_shapes

:@@*
use_locking(*
T0*$
_class
loc:@pi/dense_1/kernel
┤
save_14/Assign_16Assignpi/dense_2/biassave_14/RestoreV2:16*
use_locking(*
T0*"
_class
loc:@pi/dense_2/bias*
validate_shape(*
_output_shapes
:
╣
save_14/Assign_17Assignpi/dense_2/bias/Adamsave_14/RestoreV2:17*
use_locking(*
T0*"
_class
loc:@pi/dense_2/bias*
validate_shape(*
_output_shapes
:
╗
save_14/Assign_18Assignpi/dense_2/bias/Adam_1save_14/RestoreV2:18*
use_locking(*
T0*"
_class
loc:@pi/dense_2/bias*
validate_shape(*
_output_shapes
:
╝
save_14/Assign_19Assignpi/dense_2/kernelsave_14/RestoreV2:19*
use_locking(*
T0*$
_class
loc:@pi/dense_2/kernel*
validate_shape(*
_output_shapes

:@
┴
save_14/Assign_20Assignpi/dense_2/kernel/Adamsave_14/RestoreV2:20*
use_locking(*
T0*$
_class
loc:@pi/dense_2/kernel*
validate_shape(*
_output_shapes

:@
├
save_14/Assign_21Assignpi/dense_2/kernel/Adam_1save_14/RestoreV2:21*$
_class
loc:@pi/dense_2/kernel*
validate_shape(*
_output_shapes

:@*
use_locking(*
T0
«
save_14/Assign_22Assignv/dense/biassave_14/RestoreV2:22*
use_locking(*
T0*
_class
loc:@v/dense/bias*
validate_shape(*
_output_shapes
:@
│
save_14/Assign_23Assignv/dense/bias/Adamsave_14/RestoreV2:23*
_output_shapes
:@*
use_locking(*
T0*
_class
loc:@v/dense/bias*
validate_shape(
х
save_14/Assign_24Assignv/dense/bias/Adam_1save_14/RestoreV2:24*
validate_shape(*
_output_shapes
:@*
use_locking(*
T0*
_class
loc:@v/dense/bias
и
save_14/Assign_25Assignv/dense/kernelsave_14/RestoreV2:25*
_output_shapes
:	Є@*
use_locking(*
T0*!
_class
loc:@v/dense/kernel*
validate_shape(
╝
save_14/Assign_26Assignv/dense/kernel/Adamsave_14/RestoreV2:26*
validate_shape(*
_output_shapes
:	Є@*
use_locking(*
T0*!
_class
loc:@v/dense/kernel
Й
save_14/Assign_27Assignv/dense/kernel/Adam_1save_14/RestoreV2:27*
_output_shapes
:	Є@*
use_locking(*
T0*!
_class
loc:@v/dense/kernel*
validate_shape(
▓
save_14/Assign_28Assignv/dense_1/biassave_14/RestoreV2:28*
use_locking(*
T0*!
_class
loc:@v/dense_1/bias*
validate_shape(*
_output_shapes
:@
и
save_14/Assign_29Assignv/dense_1/bias/Adamsave_14/RestoreV2:29*!
_class
loc:@v/dense_1/bias*
validate_shape(*
_output_shapes
:@*
use_locking(*
T0
╣
save_14/Assign_30Assignv/dense_1/bias/Adam_1save_14/RestoreV2:30*
_output_shapes
:@*
use_locking(*
T0*!
_class
loc:@v/dense_1/bias*
validate_shape(
║
save_14/Assign_31Assignv/dense_1/kernelsave_14/RestoreV2:31*
use_locking(*
T0*#
_class
loc:@v/dense_1/kernel*
validate_shape(*
_output_shapes

:@@
┐
save_14/Assign_32Assignv/dense_1/kernel/Adamsave_14/RestoreV2:32*
use_locking(*
T0*#
_class
loc:@v/dense_1/kernel*
validate_shape(*
_output_shapes

:@@
┴
save_14/Assign_33Assignv/dense_1/kernel/Adam_1save_14/RestoreV2:33*
validate_shape(*
_output_shapes

:@@*
use_locking(*
T0*#
_class
loc:@v/dense_1/kernel
▓
save_14/Assign_34Assignv/dense_2/biassave_14/RestoreV2:34*
_output_shapes
:*
use_locking(*
T0*!
_class
loc:@v/dense_2/bias*
validate_shape(
и
save_14/Assign_35Assignv/dense_2/bias/Adamsave_14/RestoreV2:35*
use_locking(*
T0*!
_class
loc:@v/dense_2/bias*
validate_shape(*
_output_shapes
:
╣
save_14/Assign_36Assignv/dense_2/bias/Adam_1save_14/RestoreV2:36*
use_locking(*
T0*!
_class
loc:@v/dense_2/bias*
validate_shape(*
_output_shapes
:
║
save_14/Assign_37Assignv/dense_2/kernelsave_14/RestoreV2:37*
use_locking(*
T0*#
_class
loc:@v/dense_2/kernel*
validate_shape(*
_output_shapes

:@
┐
save_14/Assign_38Assignv/dense_2/kernel/Adamsave_14/RestoreV2:38*
T0*#
_class
loc:@v/dense_2/kernel*
validate_shape(*
_output_shapes

:@*
use_locking(
┴
save_14/Assign_39Assignv/dense_2/kernel/Adam_1save_14/RestoreV2:39*
use_locking(*
T0*#
_class
loc:@v/dense_2/kernel*
validate_shape(*
_output_shapes

:@
▒
save_14/restore_shardNoOp^save_14/Assign^save_14/Assign_1^save_14/Assign_10^save_14/Assign_11^save_14/Assign_12^save_14/Assign_13^save_14/Assign_14^save_14/Assign_15^save_14/Assign_16^save_14/Assign_17^save_14/Assign_18^save_14/Assign_19^save_14/Assign_2^save_14/Assign_20^save_14/Assign_21^save_14/Assign_22^save_14/Assign_23^save_14/Assign_24^save_14/Assign_25^save_14/Assign_26^save_14/Assign_27^save_14/Assign_28^save_14/Assign_29^save_14/Assign_3^save_14/Assign_30^save_14/Assign_31^save_14/Assign_32^save_14/Assign_33^save_14/Assign_34^save_14/Assign_35^save_14/Assign_36^save_14/Assign_37^save_14/Assign_38^save_14/Assign_39^save_14/Assign_4^save_14/Assign_5^save_14/Assign_6^save_14/Assign_7^save_14/Assign_8^save_14/Assign_9
3
save_14/restore_allNoOp^save_14/restore_shard
\
save_15/filename/inputConst*
dtype0*
_output_shapes
: *
valueB Bmodel
t
save_15/filenamePlaceholderWithDefaultsave_15/filename/input*
dtype0*
_output_shapes
: *
shape: 
k
save_15/ConstPlaceholderWithDefaultsave_15/filename*
shape: *
dtype0*
_output_shapes
: 
Є
save_15/StringJoin/inputs_1Const*<
value3B1 B+_temp_c0edf74ea26d47ea847925d7f02f1909/part*
dtype0*
_output_shapes
: 
~
save_15/StringJoin
StringJoinsave_15/Constsave_15/StringJoin/inputs_1*
N*
_output_shapes
: *
	separator 
T
save_15/num_shardsConst*
_output_shapes
: *
value	B :*
dtype0
_
save_15/ShardedFilename/shardConst*
value	B : *
dtype0*
_output_shapes
: 
Ѕ
save_15/ShardedFilenameShardedFilenamesave_15/StringJoinsave_15/ShardedFilename/shardsave_15/num_shards*
_output_shapes
: 
І
save_15/SaveV2/tensor_namesConst*╗
value▒B«(Bbeta1_powerBbeta1_power_1Bbeta2_powerBbeta2_power_1Bpi/dense/biasBpi/dense/bias/AdamBpi/dense/bias/Adam_1Bpi/dense/kernelBpi/dense/kernel/AdamBpi/dense/kernel/Adam_1Bpi/dense_1/biasBpi/dense_1/bias/AdamBpi/dense_1/bias/Adam_1Bpi/dense_1/kernelBpi/dense_1/kernel/AdamBpi/dense_1/kernel/Adam_1Bpi/dense_2/biasBpi/dense_2/bias/AdamBpi/dense_2/bias/Adam_1Bpi/dense_2/kernelBpi/dense_2/kernel/AdamBpi/dense_2/kernel/Adam_1Bv/dense/biasBv/dense/bias/AdamBv/dense/bias/Adam_1Bv/dense/kernelBv/dense/kernel/AdamBv/dense/kernel/Adam_1Bv/dense_1/biasBv/dense_1/bias/AdamBv/dense_1/bias/Adam_1Bv/dense_1/kernelBv/dense_1/kernel/AdamBv/dense_1/kernel/Adam_1Bv/dense_2/biasBv/dense_2/bias/AdamBv/dense_2/bias/Adam_1Bv/dense_2/kernelBv/dense_2/kernel/AdamBv/dense_2/kernel/Adam_1*
dtype0*
_output_shapes
:(
Х
save_15/SaveV2/shape_and_slicesConst*c
valueZBX(B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B *
dtype0*
_output_shapes
:(
═
save_15/SaveV2SaveV2save_15/ShardedFilenamesave_15/SaveV2/tensor_namessave_15/SaveV2/shape_and_slicesbeta1_powerbeta1_power_1beta2_powerbeta2_power_1pi/dense/biaspi/dense/bias/Adampi/dense/bias/Adam_1pi/dense/kernelpi/dense/kernel/Adampi/dense/kernel/Adam_1pi/dense_1/biaspi/dense_1/bias/Adampi/dense_1/bias/Adam_1pi/dense_1/kernelpi/dense_1/kernel/Adampi/dense_1/kernel/Adam_1pi/dense_2/biaspi/dense_2/bias/Adampi/dense_2/bias/Adam_1pi/dense_2/kernelpi/dense_2/kernel/Adampi/dense_2/kernel/Adam_1v/dense/biasv/dense/bias/Adamv/dense/bias/Adam_1v/dense/kernelv/dense/kernel/Adamv/dense/kernel/Adam_1v/dense_1/biasv/dense_1/bias/Adamv/dense_1/bias/Adam_1v/dense_1/kernelv/dense_1/kernel/Adamv/dense_1/kernel/Adam_1v/dense_2/biasv/dense_2/bias/Adamv/dense_2/bias/Adam_1v/dense_2/kernelv/dense_2/kernel/Adamv/dense_2/kernel/Adam_1*6
dtypes,
*2(
Ю
save_15/control_dependencyIdentitysave_15/ShardedFilename^save_15/SaveV2*
T0**
_class 
loc:@save_15/ShardedFilename*
_output_shapes
: 
д
.save_15/MergeV2Checkpoints/checkpoint_prefixesPacksave_15/ShardedFilename^save_15/control_dependency*
T0*

axis *
N*
_output_shapes
:
є
save_15/MergeV2CheckpointsMergeV2Checkpoints.save_15/MergeV2Checkpoints/checkpoint_prefixessave_15/Const*
delete_old_dirs(
є
save_15/IdentityIdentitysave_15/Const^save_15/MergeV2Checkpoints^save_15/control_dependency*
_output_shapes
: *
T0
ј
save_15/RestoreV2/tensor_namesConst*
_output_shapes
:(*╗
value▒B«(Bbeta1_powerBbeta1_power_1Bbeta2_powerBbeta2_power_1Bpi/dense/biasBpi/dense/bias/AdamBpi/dense/bias/Adam_1Bpi/dense/kernelBpi/dense/kernel/AdamBpi/dense/kernel/Adam_1Bpi/dense_1/biasBpi/dense_1/bias/AdamBpi/dense_1/bias/Adam_1Bpi/dense_1/kernelBpi/dense_1/kernel/AdamBpi/dense_1/kernel/Adam_1Bpi/dense_2/biasBpi/dense_2/bias/AdamBpi/dense_2/bias/Adam_1Bpi/dense_2/kernelBpi/dense_2/kernel/AdamBpi/dense_2/kernel/Adam_1Bv/dense/biasBv/dense/bias/AdamBv/dense/bias/Adam_1Bv/dense/kernelBv/dense/kernel/AdamBv/dense/kernel/Adam_1Bv/dense_1/biasBv/dense_1/bias/AdamBv/dense_1/bias/Adam_1Bv/dense_1/kernelBv/dense_1/kernel/AdamBv/dense_1/kernel/Adam_1Bv/dense_2/biasBv/dense_2/bias/AdamBv/dense_2/bias/Adam_1Bv/dense_2/kernelBv/dense_2/kernel/AdamBv/dense_2/kernel/Adam_1*
dtype0
╣
"save_15/RestoreV2/shape_and_slicesConst*c
valueZBX(B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B *
dtype0*
_output_shapes
:(
Р
save_15/RestoreV2	RestoreV2save_15/Constsave_15/RestoreV2/tensor_names"save_15/RestoreV2/shape_and_slices*Х
_output_shapesБ
а::::::::::::::::::::::::::::::::::::::::*6
dtypes,
*2(
ц
save_15/AssignAssignbeta1_powersave_15/RestoreV2*
validate_shape(*
_output_shapes
: *
use_locking(*
T0* 
_class
loc:@pi/dense/bias
Е
save_15/Assign_1Assignbeta1_power_1save_15/RestoreV2:1*
use_locking(*
T0*
_class
loc:@v/dense/bias*
validate_shape(*
_output_shapes
: 
е
save_15/Assign_2Assignbeta2_powersave_15/RestoreV2:2*
use_locking(*
T0* 
_class
loc:@pi/dense/bias*
validate_shape(*
_output_shapes
: 
Е
save_15/Assign_3Assignbeta2_power_1save_15/RestoreV2:3*
use_locking(*
T0*
_class
loc:@v/dense/bias*
validate_shape(*
_output_shapes
: 
«
save_15/Assign_4Assignpi/dense/biassave_15/RestoreV2:4*
use_locking(*
T0* 
_class
loc:@pi/dense/bias*
validate_shape(*
_output_shapes
:@
│
save_15/Assign_5Assignpi/dense/bias/Adamsave_15/RestoreV2:5*
use_locking(*
T0* 
_class
loc:@pi/dense/bias*
validate_shape(*
_output_shapes
:@
х
save_15/Assign_6Assignpi/dense/bias/Adam_1save_15/RestoreV2:6*
use_locking(*
T0* 
_class
loc:@pi/dense/bias*
validate_shape(*
_output_shapes
:@
и
save_15/Assign_7Assignpi/dense/kernelsave_15/RestoreV2:7*
_output_shapes
:	Є@*
use_locking(*
T0*"
_class
loc:@pi/dense/kernel*
validate_shape(
╝
save_15/Assign_8Assignpi/dense/kernel/Adamsave_15/RestoreV2:8*
use_locking(*
T0*"
_class
loc:@pi/dense/kernel*
validate_shape(*
_output_shapes
:	Є@
Й
save_15/Assign_9Assignpi/dense/kernel/Adam_1save_15/RestoreV2:9*
_output_shapes
:	Є@*
use_locking(*
T0*"
_class
loc:@pi/dense/kernel*
validate_shape(
┤
save_15/Assign_10Assignpi/dense_1/biassave_15/RestoreV2:10*
use_locking(*
T0*"
_class
loc:@pi/dense_1/bias*
validate_shape(*
_output_shapes
:@
╣
save_15/Assign_11Assignpi/dense_1/bias/Adamsave_15/RestoreV2:11*
validate_shape(*
_output_shapes
:@*
use_locking(*
T0*"
_class
loc:@pi/dense_1/bias
╗
save_15/Assign_12Assignpi/dense_1/bias/Adam_1save_15/RestoreV2:12*
use_locking(*
T0*"
_class
loc:@pi/dense_1/bias*
validate_shape(*
_output_shapes
:@
╝
save_15/Assign_13Assignpi/dense_1/kernelsave_15/RestoreV2:13*
use_locking(*
T0*$
_class
loc:@pi/dense_1/kernel*
validate_shape(*
_output_shapes

:@@
┴
save_15/Assign_14Assignpi/dense_1/kernel/Adamsave_15/RestoreV2:14*
use_locking(*
T0*$
_class
loc:@pi/dense_1/kernel*
validate_shape(*
_output_shapes

:@@
├
save_15/Assign_15Assignpi/dense_1/kernel/Adam_1save_15/RestoreV2:15*$
_class
loc:@pi/dense_1/kernel*
validate_shape(*
_output_shapes

:@@*
use_locking(*
T0
┤
save_15/Assign_16Assignpi/dense_2/biassave_15/RestoreV2:16*
use_locking(*
T0*"
_class
loc:@pi/dense_2/bias*
validate_shape(*
_output_shapes
:
╣
save_15/Assign_17Assignpi/dense_2/bias/Adamsave_15/RestoreV2:17*
_output_shapes
:*
use_locking(*
T0*"
_class
loc:@pi/dense_2/bias*
validate_shape(
╗
save_15/Assign_18Assignpi/dense_2/bias/Adam_1save_15/RestoreV2:18*
use_locking(*
T0*"
_class
loc:@pi/dense_2/bias*
validate_shape(*
_output_shapes
:
╝
save_15/Assign_19Assignpi/dense_2/kernelsave_15/RestoreV2:19*
T0*$
_class
loc:@pi/dense_2/kernel*
validate_shape(*
_output_shapes

:@*
use_locking(
┴
save_15/Assign_20Assignpi/dense_2/kernel/Adamsave_15/RestoreV2:20*
_output_shapes

:@*
use_locking(*
T0*$
_class
loc:@pi/dense_2/kernel*
validate_shape(
├
save_15/Assign_21Assignpi/dense_2/kernel/Adam_1save_15/RestoreV2:21*
validate_shape(*
_output_shapes

:@*
use_locking(*
T0*$
_class
loc:@pi/dense_2/kernel
«
save_15/Assign_22Assignv/dense/biassave_15/RestoreV2:22*
T0*
_class
loc:@v/dense/bias*
validate_shape(*
_output_shapes
:@*
use_locking(
│
save_15/Assign_23Assignv/dense/bias/Adamsave_15/RestoreV2:23*
use_locking(*
T0*
_class
loc:@v/dense/bias*
validate_shape(*
_output_shapes
:@
х
save_15/Assign_24Assignv/dense/bias/Adam_1save_15/RestoreV2:24*
use_locking(*
T0*
_class
loc:@v/dense/bias*
validate_shape(*
_output_shapes
:@
и
save_15/Assign_25Assignv/dense/kernelsave_15/RestoreV2:25*
use_locking(*
T0*!
_class
loc:@v/dense/kernel*
validate_shape(*
_output_shapes
:	Є@
╝
save_15/Assign_26Assignv/dense/kernel/Adamsave_15/RestoreV2:26*
use_locking(*
T0*!
_class
loc:@v/dense/kernel*
validate_shape(*
_output_shapes
:	Є@
Й
save_15/Assign_27Assignv/dense/kernel/Adam_1save_15/RestoreV2:27*
_output_shapes
:	Є@*
use_locking(*
T0*!
_class
loc:@v/dense/kernel*
validate_shape(
▓
save_15/Assign_28Assignv/dense_1/biassave_15/RestoreV2:28*
validate_shape(*
_output_shapes
:@*
use_locking(*
T0*!
_class
loc:@v/dense_1/bias
и
save_15/Assign_29Assignv/dense_1/bias/Adamsave_15/RestoreV2:29*
validate_shape(*
_output_shapes
:@*
use_locking(*
T0*!
_class
loc:@v/dense_1/bias
╣
save_15/Assign_30Assignv/dense_1/bias/Adam_1save_15/RestoreV2:30*
use_locking(*
T0*!
_class
loc:@v/dense_1/bias*
validate_shape(*
_output_shapes
:@
║
save_15/Assign_31Assignv/dense_1/kernelsave_15/RestoreV2:31*
validate_shape(*
_output_shapes

:@@*
use_locking(*
T0*#
_class
loc:@v/dense_1/kernel
┐
save_15/Assign_32Assignv/dense_1/kernel/Adamsave_15/RestoreV2:32*
_output_shapes

:@@*
use_locking(*
T0*#
_class
loc:@v/dense_1/kernel*
validate_shape(
┴
save_15/Assign_33Assignv/dense_1/kernel/Adam_1save_15/RestoreV2:33*
use_locking(*
T0*#
_class
loc:@v/dense_1/kernel*
validate_shape(*
_output_shapes

:@@
▓
save_15/Assign_34Assignv/dense_2/biassave_15/RestoreV2:34*
use_locking(*
T0*!
_class
loc:@v/dense_2/bias*
validate_shape(*
_output_shapes
:
и
save_15/Assign_35Assignv/dense_2/bias/Adamsave_15/RestoreV2:35*
use_locking(*
T0*!
_class
loc:@v/dense_2/bias*
validate_shape(*
_output_shapes
:
╣
save_15/Assign_36Assignv/dense_2/bias/Adam_1save_15/RestoreV2:36*
use_locking(*
T0*!
_class
loc:@v/dense_2/bias*
validate_shape(*
_output_shapes
:
║
save_15/Assign_37Assignv/dense_2/kernelsave_15/RestoreV2:37*
_output_shapes

:@*
use_locking(*
T0*#
_class
loc:@v/dense_2/kernel*
validate_shape(
┐
save_15/Assign_38Assignv/dense_2/kernel/Adamsave_15/RestoreV2:38*
validate_shape(*
_output_shapes

:@*
use_locking(*
T0*#
_class
loc:@v/dense_2/kernel
┴
save_15/Assign_39Assignv/dense_2/kernel/Adam_1save_15/RestoreV2:39*
use_locking(*
T0*#
_class
loc:@v/dense_2/kernel*
validate_shape(*
_output_shapes

:@
▒
save_15/restore_shardNoOp^save_15/Assign^save_15/Assign_1^save_15/Assign_10^save_15/Assign_11^save_15/Assign_12^save_15/Assign_13^save_15/Assign_14^save_15/Assign_15^save_15/Assign_16^save_15/Assign_17^save_15/Assign_18^save_15/Assign_19^save_15/Assign_2^save_15/Assign_20^save_15/Assign_21^save_15/Assign_22^save_15/Assign_23^save_15/Assign_24^save_15/Assign_25^save_15/Assign_26^save_15/Assign_27^save_15/Assign_28^save_15/Assign_29^save_15/Assign_3^save_15/Assign_30^save_15/Assign_31^save_15/Assign_32^save_15/Assign_33^save_15/Assign_34^save_15/Assign_35^save_15/Assign_36^save_15/Assign_37^save_15/Assign_38^save_15/Assign_39^save_15/Assign_4^save_15/Assign_5^save_15/Assign_6^save_15/Assign_7^save_15/Assign_8^save_15/Assign_9
3
save_15/restore_allNoOp^save_15/restore_shard "єE
save_15/Const:0save_15/Identity:0save_15/restore_all (5 @F8"
train_op

Adam
Adam_1"т%
	variablesО%н%
s
pi/dense/kernel:0pi/dense/kernel/Assignpi/dense/kernel/read:02,pi/dense/kernel/Initializer/random_uniform:08
b
pi/dense/bias:0pi/dense/bias/Assignpi/dense/bias/read:02!pi/dense/bias/Initializer/zeros:08
{
pi/dense_1/kernel:0pi/dense_1/kernel/Assignpi/dense_1/kernel/read:02.pi/dense_1/kernel/Initializer/random_uniform:08
j
pi/dense_1/bias:0pi/dense_1/bias/Assignpi/dense_1/bias/read:02#pi/dense_1/bias/Initializer/zeros:08
{
pi/dense_2/kernel:0pi/dense_2/kernel/Assignpi/dense_2/kernel/read:02.pi/dense_2/kernel/Initializer/random_uniform:08
j
pi/dense_2/bias:0pi/dense_2/bias/Assignpi/dense_2/bias/read:02#pi/dense_2/bias/Initializer/zeros:08
o
v/dense/kernel:0v/dense/kernel/Assignv/dense/kernel/read:02+v/dense/kernel/Initializer/random_uniform:08
^
v/dense/bias:0v/dense/bias/Assignv/dense/bias/read:02 v/dense/bias/Initializer/zeros:08
w
v/dense_1/kernel:0v/dense_1/kernel/Assignv/dense_1/kernel/read:02-v/dense_1/kernel/Initializer/random_uniform:08
f
v/dense_1/bias:0v/dense_1/bias/Assignv/dense_1/bias/read:02"v/dense_1/bias/Initializer/zeros:08
w
v/dense_2/kernel:0v/dense_2/kernel/Assignv/dense_2/kernel/read:02-v/dense_2/kernel/Initializer/random_uniform:08
f
v/dense_2/bias:0v/dense_2/bias/Assignv/dense_2/bias/read:02"v/dense_2/bias/Initializer/zeros:08
T
beta1_power:0beta1_power/Assignbeta1_power/read:02beta1_power/initial_value:0
T
beta2_power:0beta2_power/Assignbeta2_power/read:02beta2_power/initial_value:0
|
pi/dense/kernel/Adam:0pi/dense/kernel/Adam/Assignpi/dense/kernel/Adam/read:02(pi/dense/kernel/Adam/Initializer/zeros:0
ё
pi/dense/kernel/Adam_1:0pi/dense/kernel/Adam_1/Assignpi/dense/kernel/Adam_1/read:02*pi/dense/kernel/Adam_1/Initializer/zeros:0
t
pi/dense/bias/Adam:0pi/dense/bias/Adam/Assignpi/dense/bias/Adam/read:02&pi/dense/bias/Adam/Initializer/zeros:0
|
pi/dense/bias/Adam_1:0pi/dense/bias/Adam_1/Assignpi/dense/bias/Adam_1/read:02(pi/dense/bias/Adam_1/Initializer/zeros:0
ё
pi/dense_1/kernel/Adam:0pi/dense_1/kernel/Adam/Assignpi/dense_1/kernel/Adam/read:02*pi/dense_1/kernel/Adam/Initializer/zeros:0
ї
pi/dense_1/kernel/Adam_1:0pi/dense_1/kernel/Adam_1/Assignpi/dense_1/kernel/Adam_1/read:02,pi/dense_1/kernel/Adam_1/Initializer/zeros:0
|
pi/dense_1/bias/Adam:0pi/dense_1/bias/Adam/Assignpi/dense_1/bias/Adam/read:02(pi/dense_1/bias/Adam/Initializer/zeros:0
ё
pi/dense_1/bias/Adam_1:0pi/dense_1/bias/Adam_1/Assignpi/dense_1/bias/Adam_1/read:02*pi/dense_1/bias/Adam_1/Initializer/zeros:0
ё
pi/dense_2/kernel/Adam:0pi/dense_2/kernel/Adam/Assignpi/dense_2/kernel/Adam/read:02*pi/dense_2/kernel/Adam/Initializer/zeros:0
ї
pi/dense_2/kernel/Adam_1:0pi/dense_2/kernel/Adam_1/Assignpi/dense_2/kernel/Adam_1/read:02,pi/dense_2/kernel/Adam_1/Initializer/zeros:0
|
pi/dense_2/bias/Adam:0pi/dense_2/bias/Adam/Assignpi/dense_2/bias/Adam/read:02(pi/dense_2/bias/Adam/Initializer/zeros:0
ё
pi/dense_2/bias/Adam_1:0pi/dense_2/bias/Adam_1/Assignpi/dense_2/bias/Adam_1/read:02*pi/dense_2/bias/Adam_1/Initializer/zeros:0
\
beta1_power_1:0beta1_power_1/Assignbeta1_power_1/read:02beta1_power_1/initial_value:0
\
beta2_power_1:0beta2_power_1/Assignbeta2_power_1/read:02beta2_power_1/initial_value:0
x
v/dense/kernel/Adam:0v/dense/kernel/Adam/Assignv/dense/kernel/Adam/read:02'v/dense/kernel/Adam/Initializer/zeros:0
ђ
v/dense/kernel/Adam_1:0v/dense/kernel/Adam_1/Assignv/dense/kernel/Adam_1/read:02)v/dense/kernel/Adam_1/Initializer/zeros:0
p
v/dense/bias/Adam:0v/dense/bias/Adam/Assignv/dense/bias/Adam/read:02%v/dense/bias/Adam/Initializer/zeros:0
x
v/dense/bias/Adam_1:0v/dense/bias/Adam_1/Assignv/dense/bias/Adam_1/read:02'v/dense/bias/Adam_1/Initializer/zeros:0
ђ
v/dense_1/kernel/Adam:0v/dense_1/kernel/Adam/Assignv/dense_1/kernel/Adam/read:02)v/dense_1/kernel/Adam/Initializer/zeros:0
ѕ
v/dense_1/kernel/Adam_1:0v/dense_1/kernel/Adam_1/Assignv/dense_1/kernel/Adam_1/read:02+v/dense_1/kernel/Adam_1/Initializer/zeros:0
x
v/dense_1/bias/Adam:0v/dense_1/bias/Adam/Assignv/dense_1/bias/Adam/read:02'v/dense_1/bias/Adam/Initializer/zeros:0
ђ
v/dense_1/bias/Adam_1:0v/dense_1/bias/Adam_1/Assignv/dense_1/bias/Adam_1/read:02)v/dense_1/bias/Adam_1/Initializer/zeros:0
ђ
v/dense_2/kernel/Adam:0v/dense_2/kernel/Adam/Assignv/dense_2/kernel/Adam/read:02)v/dense_2/kernel/Adam/Initializer/zeros:0
ѕ
v/dense_2/kernel/Adam_1:0v/dense_2/kernel/Adam_1/Assignv/dense_2/kernel/Adam_1/read:02+v/dense_2/kernel/Adam_1/Initializer/zeros:0
x
v/dense_2/bias/Adam:0v/dense_2/bias/Adam/Assignv/dense_2/bias/Adam/read:02'v/dense_2/bias/Adam/Initializer/zeros:0
ђ
v/dense_2/bias/Adam_1:0v/dense_2/bias/Adam_1/Assignv/dense_2/bias/Adam_1/read:02)v/dense_2/bias/Adam_1/Initializer/zeros:0"┘

trainable_variables┴
Й

s
pi/dense/kernel:0pi/dense/kernel/Assignpi/dense/kernel/read:02,pi/dense/kernel/Initializer/random_uniform:08
b
pi/dense/bias:0pi/dense/bias/Assignpi/dense/bias/read:02!pi/dense/bias/Initializer/zeros:08
{
pi/dense_1/kernel:0pi/dense_1/kernel/Assignpi/dense_1/kernel/read:02.pi/dense_1/kernel/Initializer/random_uniform:08
j
pi/dense_1/bias:0pi/dense_1/bias/Assignpi/dense_1/bias/read:02#pi/dense_1/bias/Initializer/zeros:08
{
pi/dense_2/kernel:0pi/dense_2/kernel/Assignpi/dense_2/kernel/read:02.pi/dense_2/kernel/Initializer/random_uniform:08
j
pi/dense_2/bias:0pi/dense_2/bias/Assignpi/dense_2/bias/read:02#pi/dense_2/bias/Initializer/zeros:08
o
v/dense/kernel:0v/dense/kernel/Assignv/dense/kernel/read:02+v/dense/kernel/Initializer/random_uniform:08
^
v/dense/bias:0v/dense/bias/Assignv/dense/bias/read:02 v/dense/bias/Initializer/zeros:08
w
v/dense_1/kernel:0v/dense_1/kernel/Assignv/dense_1/kernel/read:02-v/dense_1/kernel/Initializer/random_uniform:08
f
v/dense_1/bias:0v/dense_1/bias/Assignv/dense_1/bias/read:02"v/dense_1/bias/Initializer/zeros:08
w
v/dense_2/kernel:0v/dense_2/kernel/Assignv/dense_2/kernel/read:02-v/dense_2/kernel/Initializer/random_uniform:08
f
v/dense_2/bias:0v/dense_2/bias/Assignv/dense_2/bias/read:02"v/dense_2/bias/Initializer/zeros:08*е
serving_defaultћ
*
x%
Placeholder:0         Є#
v
v/Squeeze:0         %
pi
pi/Squeeze:0	         tensorflow/serving/predict