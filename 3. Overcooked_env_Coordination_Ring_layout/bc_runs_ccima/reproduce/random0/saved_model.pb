ܝ
��
^
AssignVariableOp
resource
value"dtype"
dtypetype"
validate_shapebool( �
~
BiasAdd

value"T	
bias"T
output"T" 
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
8
Const
output"dtype"
valuetensor"
dtypetype
.
Identity

input"T
output"T"	
Ttype
q
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:

2	
�
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool("
allow_missing_filesbool( �

NoOp
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
@
ReadVariableOp
resource
value"dtype"
dtypetype�
E
Relu
features"T
activations"T"
Ttype:
2	
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0�
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0�
?
Select
	condition

t"T
e"T
output"T"	
Ttype
H
ShardedFilename
basename	
shard

num_shards
filename
�
StatefulPartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring ��
@
StaticRegexFullMatch	
input

output
"
patternstring
N

StringJoin
inputs*N

output"
Nint(0"
	separatorstring 
�
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 �"serve*2.11.02v2.11.0-rc2-17-gd5b57ca93e58ؚ
^
countVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namecount
W
count/Read/ReadVariableOpReadVariableOpcount*
_output_shapes
: *
dtype0
^
totalVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nametotal
W
total/Read/ReadVariableOpReadVariableOptotal*
_output_shapes
: *
dtype0
b
count_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	count_1
[
count_1/Read/ReadVariableOpReadVariableOpcount_1*
_output_shapes
: *
dtype0
b
total_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	total_1
[
total_1/Read/ReadVariableOpReadVariableOptotal_1*
_output_shapes
: *
dtype0
|
Adam/v/logits/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*#
shared_nameAdam/v/logits/bias
u
&Adam/v/logits/bias/Read/ReadVariableOpReadVariableOpAdam/v/logits/bias*
_output_shapes
:*
dtype0
|
Adam/m/logits/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*#
shared_nameAdam/m/logits/bias
u
&Adam/m/logits/bias/Read/ReadVariableOpReadVariableOpAdam/m/logits/bias*
_output_shapes
:*
dtype0
�
Adam/v/logits/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@*%
shared_nameAdam/v/logits/kernel
}
(Adam/v/logits/kernel/Read/ReadVariableOpReadVariableOpAdam/v/logits/kernel*
_output_shapes

:@*
dtype0
�
Adam/m/logits/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@*%
shared_nameAdam/m/logits/kernel
}
(Adam/m/logits/kernel/Read/ReadVariableOpReadVariableOpAdam/m/logits/kernel*
_output_shapes

:@*
dtype0
x
Adam/v/fc_1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*!
shared_nameAdam/v/fc_1/bias
q
$Adam/v/fc_1/bias/Read/ReadVariableOpReadVariableOpAdam/v/fc_1/bias*
_output_shapes
:@*
dtype0
x
Adam/m/fc_1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*!
shared_nameAdam/m/fc_1/bias
q
$Adam/m/fc_1/bias/Read/ReadVariableOpReadVariableOpAdam/m/fc_1/bias*
_output_shapes
:@*
dtype0
�
Adam/v/fc_1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@@*#
shared_nameAdam/v/fc_1/kernel
y
&Adam/v/fc_1/kernel/Read/ReadVariableOpReadVariableOpAdam/v/fc_1/kernel*
_output_shapes

:@@*
dtype0
�
Adam/m/fc_1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@@*#
shared_nameAdam/m/fc_1/kernel
y
&Adam/m/fc_1/kernel/Read/ReadVariableOpReadVariableOpAdam/m/fc_1/kernel*
_output_shapes

:@@*
dtype0
x
Adam/v/fc_0/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*!
shared_nameAdam/v/fc_0/bias
q
$Adam/v/fc_0/bias/Read/ReadVariableOpReadVariableOpAdam/v/fc_0/bias*
_output_shapes
:@*
dtype0
x
Adam/m/fc_0/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*!
shared_nameAdam/m/fc_0/bias
q
$Adam/m/fc_0/bias/Read/ReadVariableOpReadVariableOpAdam/m/fc_0/bias*
_output_shapes
:@*
dtype0
�
Adam/v/fc_0/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:`@*#
shared_nameAdam/v/fc_0/kernel
y
&Adam/v/fc_0/kernel/Read/ReadVariableOpReadVariableOpAdam/v/fc_0/kernel*
_output_shapes

:`@*
dtype0
�
Adam/m/fc_0/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:`@*#
shared_nameAdam/m/fc_0/kernel
y
&Adam/m/fc_0/kernel/Read/ReadVariableOpReadVariableOpAdam/m/fc_0/kernel*
_output_shapes

:`@*
dtype0
n
learning_rateVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namelearning_rate
g
!learning_rate/Read/ReadVariableOpReadVariableOplearning_rate*
_output_shapes
: *
dtype0
f
	iterationVarHandleOp*
_output_shapes
: *
dtype0	*
shape: *
shared_name	iteration
_
iteration/Read/ReadVariableOpReadVariableOp	iteration*
_output_shapes
: *
dtype0	
n
logits/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namelogits/bias
g
logits/bias/Read/ReadVariableOpReadVariableOplogits/bias*
_output_shapes
:*
dtype0
v
logits/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@*
shared_namelogits/kernel
o
!logits/kernel/Read/ReadVariableOpReadVariableOplogits/kernel*
_output_shapes

:@*
dtype0
j
	fc_1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*
shared_name	fc_1/bias
c
fc_1/bias/Read/ReadVariableOpReadVariableOp	fc_1/bias*
_output_shapes
:@*
dtype0
r
fc_1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@@*
shared_namefc_1/kernel
k
fc_1/kernel/Read/ReadVariableOpReadVariableOpfc_1/kernel*
_output_shapes

:@@*
dtype0
j
	fc_0/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*
shared_name	fc_0/bias
c
fc_0/bias/Read/ReadVariableOpReadVariableOp	fc_0/bias*
_output_shapes
:@*
dtype0
r
fc_0/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:`@*
shared_namefc_0/kernel
k
fc_0/kernel/Read/ReadVariableOpReadVariableOpfc_0/kernel*
_output_shapes

:`@*
dtype0
�
&serving_default_Overcooked_observationPlaceholder*'
_output_shapes
:���������`*
dtype0*
shape:���������`
�
StatefulPartitionedCallStatefulPartitionedCall&serving_default_Overcooked_observationfc_0/kernel	fc_0/biasfc_1/kernel	fc_1/biaslogits/kernellogits/bias*
Tin
	2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*(
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8� *.
f)R'
%__inference_signature_wrapper_1031336

NoOpNoOp
�*
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*�*
value�*B�* B�*
�
layer-0
layer_with_weights-0
layer-1
layer_with_weights-1
layer-2
layer_with_weights-2
layer-3
	variables
trainable_variables
regularization_losses
	keras_api
	__call__
*
&call_and_return_all_conditional_losses
_default_save_signature
	optimizer

signatures*
* 
�
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses

kernel
bias*
�
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses

kernel
bias*
�
	variables
trainable_variables
 regularization_losses
!	keras_api
"__call__
*#&call_and_return_all_conditional_losses

$kernel
%bias*
.
0
1
2
3
$4
%5*
.
0
1
2
3
$4
%5*
* 
�
&non_trainable_variables

'layers
(metrics
)layer_regularization_losses
*layer_metrics
	variables
trainable_variables
regularization_losses
	__call__
_default_save_signature
*
&call_and_return_all_conditional_losses
&
"call_and_return_conditional_losses*
6
+trace_0
,trace_1
-trace_2
.trace_3* 
6
/trace_0
0trace_1
1trace_2
2trace_3* 
* 
�
3
_variables
4_iterations
5_learning_rate
6_index_dict
7
_momentums
8_velocities
9_update_step_xla*

:serving_default* 

0
1*

0
1*
* 
�
;non_trainable_variables

<layers
=metrics
>layer_regularization_losses
?layer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses*

@trace_0* 

Atrace_0* 
[U
VARIABLE_VALUEfc_0/kernel6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUE*
WQ
VARIABLE_VALUE	fc_0/bias4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUE*

0
1*

0
1*
* 
�
Bnon_trainable_variables

Clayers
Dmetrics
Elayer_regularization_losses
Flayer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses*

Gtrace_0* 

Htrace_0* 
[U
VARIABLE_VALUEfc_1/kernel6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUE*
WQ
VARIABLE_VALUE	fc_1/bias4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUE*

$0
%1*

$0
%1*
* 
�
Inon_trainable_variables

Jlayers
Kmetrics
Llayer_regularization_losses
Mlayer_metrics
	variables
trainable_variables
 regularization_losses
"__call__
*#&call_and_return_all_conditional_losses
&#"call_and_return_conditional_losses*

Ntrace_0* 

Otrace_0* 
]W
VARIABLE_VALUElogits/kernel6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE*
YS
VARIABLE_VALUElogits/bias4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 
 
0
1
2
3*

P0
Q1*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
b
40
R1
S2
T3
U4
V5
W6
X7
Y8
Z9
[10
\11
]12*
SM
VARIABLE_VALUE	iteration0optimizer/_iterations/.ATTRIBUTES/VARIABLE_VALUE*
ZT
VARIABLE_VALUElearning_rate3optimizer/_learning_rate/.ATTRIBUTES/VARIABLE_VALUE*
* 
.
R0
T1
V2
X3
Z4
\5*
.
S0
U1
W2
Y3
[4
]5*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
8
^	variables
_	keras_api
	`total
	acount*
H
b	variables
c	keras_api
	dtotal
	ecount
f
_fn_kwargs*
]W
VARIABLE_VALUEAdam/m/fc_0/kernel1optimizer/_variables/1/.ATTRIBUTES/VARIABLE_VALUE*
]W
VARIABLE_VALUEAdam/v/fc_0/kernel1optimizer/_variables/2/.ATTRIBUTES/VARIABLE_VALUE*
[U
VARIABLE_VALUEAdam/m/fc_0/bias1optimizer/_variables/3/.ATTRIBUTES/VARIABLE_VALUE*
[U
VARIABLE_VALUEAdam/v/fc_0/bias1optimizer/_variables/4/.ATTRIBUTES/VARIABLE_VALUE*
]W
VARIABLE_VALUEAdam/m/fc_1/kernel1optimizer/_variables/5/.ATTRIBUTES/VARIABLE_VALUE*
]W
VARIABLE_VALUEAdam/v/fc_1/kernel1optimizer/_variables/6/.ATTRIBUTES/VARIABLE_VALUE*
[U
VARIABLE_VALUEAdam/m/fc_1/bias1optimizer/_variables/7/.ATTRIBUTES/VARIABLE_VALUE*
[U
VARIABLE_VALUEAdam/v/fc_1/bias1optimizer/_variables/8/.ATTRIBUTES/VARIABLE_VALUE*
_Y
VARIABLE_VALUEAdam/m/logits/kernel1optimizer/_variables/9/.ATTRIBUTES/VARIABLE_VALUE*
`Z
VARIABLE_VALUEAdam/v/logits/kernel2optimizer/_variables/10/.ATTRIBUTES/VARIABLE_VALUE*
^X
VARIABLE_VALUEAdam/m/logits/bias2optimizer/_variables/11/.ATTRIBUTES/VARIABLE_VALUE*
^X
VARIABLE_VALUEAdam/v/logits/bias2optimizer/_variables/12/.ATTRIBUTES/VARIABLE_VALUE*

`0
a1*

^	variables*
UO
VARIABLE_VALUEtotal_14keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE*
UO
VARIABLE_VALUEcount_14keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE*

d0
e1*

b	variables*
SM
VARIABLE_VALUEtotal4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUE*
SM
VARIABLE_VALUEcount4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUE*
* 
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
�	
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filenamefc_0/kernel/Read/ReadVariableOpfc_0/bias/Read/ReadVariableOpfc_1/kernel/Read/ReadVariableOpfc_1/bias/Read/ReadVariableOp!logits/kernel/Read/ReadVariableOplogits/bias/Read/ReadVariableOpiteration/Read/ReadVariableOp!learning_rate/Read/ReadVariableOp&Adam/m/fc_0/kernel/Read/ReadVariableOp&Adam/v/fc_0/kernel/Read/ReadVariableOp$Adam/m/fc_0/bias/Read/ReadVariableOp$Adam/v/fc_0/bias/Read/ReadVariableOp&Adam/m/fc_1/kernel/Read/ReadVariableOp&Adam/v/fc_1/kernel/Read/ReadVariableOp$Adam/m/fc_1/bias/Read/ReadVariableOp$Adam/v/fc_1/bias/Read/ReadVariableOp(Adam/m/logits/kernel/Read/ReadVariableOp(Adam/v/logits/kernel/Read/ReadVariableOp&Adam/m/logits/bias/Read/ReadVariableOp&Adam/v/logits/bias/Read/ReadVariableOptotal_1/Read/ReadVariableOpcount_1/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOpConst*%
Tin
2	*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *)
f$R"
 __inference__traced_save_1031572
�
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenamefc_0/kernel	fc_0/biasfc_1/kernel	fc_1/biaslogits/kernellogits/bias	iterationlearning_rateAdam/m/fc_0/kernelAdam/v/fc_0/kernelAdam/m/fc_0/biasAdam/v/fc_0/biasAdam/m/fc_1/kernelAdam/v/fc_1/kernelAdam/m/fc_1/biasAdam/v/fc_1/biasAdam/m/logits/kernelAdam/v/logits/kernelAdam/m/logits/biasAdam/v/logits/biastotal_1count_1totalcount*$
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *,
f'R%
#__inference__traced_restore_1031654��
�
�
(__inference_logits_layer_call_fn_1031467

inputs
unknown:@
	unknown_0:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *L
fGRE
C__inference_logits_layer_call_and_return_conditional_losses_1031155o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������@: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:���������@
 
_user_specified_nameinputs
�

�
A__inference_fc_0_layer_call_and_return_conditional_losses_1031122

inputs0
matmul_readvariableop_resource:`@-
biasadd_readvariableop_resource:@
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:`@*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@P
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:���������@a
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:���������@w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������`: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:���������`
 
_user_specified_nameinputs
�
�
D__inference_model_3_layer_call_and_return_conditional_losses_1031394

inputs5
#fc_0_matmul_readvariableop_resource:`@2
$fc_0_biasadd_readvariableop_resource:@5
#fc_1_matmul_readvariableop_resource:@@2
$fc_1_biasadd_readvariableop_resource:@7
%logits_matmul_readvariableop_resource:@4
&logits_biasadd_readvariableop_resource:
identity��fc_0/BiasAdd/ReadVariableOp�fc_0/MatMul/ReadVariableOp�fc_1/BiasAdd/ReadVariableOp�fc_1/MatMul/ReadVariableOp�logits/BiasAdd/ReadVariableOp�logits/MatMul/ReadVariableOp~
fc_0/MatMul/ReadVariableOpReadVariableOp#fc_0_matmul_readvariableop_resource*
_output_shapes

:`@*
dtype0s
fc_0/MatMulMatMulinputs"fc_0/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@|
fc_0/BiasAdd/ReadVariableOpReadVariableOp$fc_0_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0�
fc_0/BiasAddBiasAddfc_0/MatMul:product:0#fc_0/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@Z
	fc_0/ReluRelufc_0/BiasAdd:output:0*
T0*'
_output_shapes
:���������@~
fc_1/MatMul/ReadVariableOpReadVariableOp#fc_1_matmul_readvariableop_resource*
_output_shapes

:@@*
dtype0�
fc_1/MatMulMatMulfc_0/Relu:activations:0"fc_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@|
fc_1/BiasAdd/ReadVariableOpReadVariableOp$fc_1_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0�
fc_1/BiasAddBiasAddfc_1/MatMul:product:0#fc_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@Z
	fc_1/ReluRelufc_1/BiasAdd:output:0*
T0*'
_output_shapes
:���������@�
logits/MatMul/ReadVariableOpReadVariableOp%logits_matmul_readvariableop_resource*
_output_shapes

:@*
dtype0�
logits/MatMulMatMulfc_1/Relu:activations:0$logits/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
logits/BiasAdd/ReadVariableOpReadVariableOp&logits_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
logits/BiasAddBiasAddlogits/MatMul:product:0%logits/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������f
IdentityIdentitylogits/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp^fc_0/BiasAdd/ReadVariableOp^fc_0/MatMul/ReadVariableOp^fc_1/BiasAdd/ReadVariableOp^fc_1/MatMul/ReadVariableOp^logits/BiasAdd/ReadVariableOp^logits/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:���������`: : : : : : 2:
fc_0/BiasAdd/ReadVariableOpfc_0/BiasAdd/ReadVariableOp28
fc_0/MatMul/ReadVariableOpfc_0/MatMul/ReadVariableOp2:
fc_1/BiasAdd/ReadVariableOpfc_1/BiasAdd/ReadVariableOp28
fc_1/MatMul/ReadVariableOpfc_1/MatMul/ReadVariableOp2>
logits/BiasAdd/ReadVariableOplogits/BiasAdd/ReadVariableOp2<
logits/MatMul/ReadVariableOplogits/MatMul/ReadVariableOp:O K
'
_output_shapes
:���������`
 
_user_specified_nameinputs
�
�
&__inference_fc_0_layer_call_fn_1031427

inputs
unknown:`@
	unknown_0:@
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *J
fERC
A__inference_fc_0_layer_call_and_return_conditional_losses_1031122o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������@`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������`: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:���������`
 
_user_specified_nameinputs
�
�
)__inference_model_3_layer_call_fn_1031370

inputs
unknown:`@
	unknown_0:@
	unknown_1:@@
	unknown_2:@
	unknown_3:@
	unknown_4:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*(
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8� *M
fHRF
D__inference_model_3_layer_call_and_return_conditional_losses_1031245o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:���������`: : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:���������`
 
_user_specified_nameinputs
�	
�
)__inference_model_3_layer_call_fn_1031277
overcooked_observation
unknown:`@
	unknown_0:@
	unknown_1:@@
	unknown_2:@
	unknown_3:@
	unknown_4:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallovercooked_observationunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*(
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8� *M
fHRF
D__inference_model_3_layer_call_and_return_conditional_losses_1031245o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:���������`: : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:_ [
'
_output_shapes
:���������`
0
_user_specified_nameOvercooked_observation
�
�
D__inference_model_3_layer_call_and_return_conditional_losses_1031315
overcooked_observation
fc_0_1031299:`@
fc_0_1031301:@
fc_1_1031304:@@
fc_1_1031306:@ 
logits_1031309:@
logits_1031311:
identity��fc_0/StatefulPartitionedCall�fc_1/StatefulPartitionedCall�logits/StatefulPartitionedCall�
fc_0/StatefulPartitionedCallStatefulPartitionedCallovercooked_observationfc_0_1031299fc_0_1031301*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *J
fERC
A__inference_fc_0_layer_call_and_return_conditional_losses_1031122�
fc_1/StatefulPartitionedCallStatefulPartitionedCall%fc_0/StatefulPartitionedCall:output:0fc_1_1031304fc_1_1031306*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *J
fERC
A__inference_fc_1_layer_call_and_return_conditional_losses_1031139�
logits/StatefulPartitionedCallStatefulPartitionedCall%fc_1/StatefulPartitionedCall:output:0logits_1031309logits_1031311*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *L
fGRE
C__inference_logits_layer_call_and_return_conditional_losses_1031155v
IdentityIdentity'logits/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp^fc_0/StatefulPartitionedCall^fc_1/StatefulPartitionedCall^logits/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:���������`: : : : : : 2<
fc_0/StatefulPartitionedCallfc_0/StatefulPartitionedCall2<
fc_1/StatefulPartitionedCallfc_1/StatefulPartitionedCall2@
logits/StatefulPartitionedCalllogits/StatefulPartitionedCall:_ [
'
_output_shapes
:���������`
0
_user_specified_nameOvercooked_observation
�	
�
C__inference_logits_layer_call_and_return_conditional_losses_1031155

inputs0
matmul_readvariableop_resource:@-
biasadd_readvariableop_resource:
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:@*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������_
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:���������w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:���������@
 
_user_specified_nameinputs
�
�
D__inference_model_3_layer_call_and_return_conditional_losses_1031245

inputs
fc_0_1031229:`@
fc_0_1031231:@
fc_1_1031234:@@
fc_1_1031236:@ 
logits_1031239:@
logits_1031241:
identity��fc_0/StatefulPartitionedCall�fc_1/StatefulPartitionedCall�logits/StatefulPartitionedCall�
fc_0/StatefulPartitionedCallStatefulPartitionedCallinputsfc_0_1031229fc_0_1031231*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *J
fERC
A__inference_fc_0_layer_call_and_return_conditional_losses_1031122�
fc_1/StatefulPartitionedCallStatefulPartitionedCall%fc_0/StatefulPartitionedCall:output:0fc_1_1031234fc_1_1031236*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *J
fERC
A__inference_fc_1_layer_call_and_return_conditional_losses_1031139�
logits/StatefulPartitionedCallStatefulPartitionedCall%fc_1/StatefulPartitionedCall:output:0logits_1031239logits_1031241*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *L
fGRE
C__inference_logits_layer_call_and_return_conditional_losses_1031155v
IdentityIdentity'logits/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp^fc_0/StatefulPartitionedCall^fc_1/StatefulPartitionedCall^logits/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:���������`: : : : : : 2<
fc_0/StatefulPartitionedCallfc_0/StatefulPartitionedCall2<
fc_1/StatefulPartitionedCallfc_1/StatefulPartitionedCall2@
logits/StatefulPartitionedCalllogits/StatefulPartitionedCall:O K
'
_output_shapes
:���������`
 
_user_specified_nameinputs
�
�
D__inference_model_3_layer_call_and_return_conditional_losses_1031418

inputs5
#fc_0_matmul_readvariableop_resource:`@2
$fc_0_biasadd_readvariableop_resource:@5
#fc_1_matmul_readvariableop_resource:@@2
$fc_1_biasadd_readvariableop_resource:@7
%logits_matmul_readvariableop_resource:@4
&logits_biasadd_readvariableop_resource:
identity��fc_0/BiasAdd/ReadVariableOp�fc_0/MatMul/ReadVariableOp�fc_1/BiasAdd/ReadVariableOp�fc_1/MatMul/ReadVariableOp�logits/BiasAdd/ReadVariableOp�logits/MatMul/ReadVariableOp~
fc_0/MatMul/ReadVariableOpReadVariableOp#fc_0_matmul_readvariableop_resource*
_output_shapes

:`@*
dtype0s
fc_0/MatMulMatMulinputs"fc_0/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@|
fc_0/BiasAdd/ReadVariableOpReadVariableOp$fc_0_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0�
fc_0/BiasAddBiasAddfc_0/MatMul:product:0#fc_0/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@Z
	fc_0/ReluRelufc_0/BiasAdd:output:0*
T0*'
_output_shapes
:���������@~
fc_1/MatMul/ReadVariableOpReadVariableOp#fc_1_matmul_readvariableop_resource*
_output_shapes

:@@*
dtype0�
fc_1/MatMulMatMulfc_0/Relu:activations:0"fc_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@|
fc_1/BiasAdd/ReadVariableOpReadVariableOp$fc_1_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0�
fc_1/BiasAddBiasAddfc_1/MatMul:product:0#fc_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@Z
	fc_1/ReluRelufc_1/BiasAdd:output:0*
T0*'
_output_shapes
:���������@�
logits/MatMul/ReadVariableOpReadVariableOp%logits_matmul_readvariableop_resource*
_output_shapes

:@*
dtype0�
logits/MatMulMatMulfc_1/Relu:activations:0$logits/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
logits/BiasAdd/ReadVariableOpReadVariableOp&logits_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
logits/BiasAddBiasAddlogits/MatMul:product:0%logits/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������f
IdentityIdentitylogits/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp^fc_0/BiasAdd/ReadVariableOp^fc_0/MatMul/ReadVariableOp^fc_1/BiasAdd/ReadVariableOp^fc_1/MatMul/ReadVariableOp^logits/BiasAdd/ReadVariableOp^logits/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:���������`: : : : : : 2:
fc_0/BiasAdd/ReadVariableOpfc_0/BiasAdd/ReadVariableOp28
fc_0/MatMul/ReadVariableOpfc_0/MatMul/ReadVariableOp2:
fc_1/BiasAdd/ReadVariableOpfc_1/BiasAdd/ReadVariableOp28
fc_1/MatMul/ReadVariableOpfc_1/MatMul/ReadVariableOp2>
logits/BiasAdd/ReadVariableOplogits/BiasAdd/ReadVariableOp2<
logits/MatMul/ReadVariableOplogits/MatMul/ReadVariableOp:O K
'
_output_shapes
:���������`
 
_user_specified_nameinputs
�

�
A__inference_fc_1_layer_call_and_return_conditional_losses_1031139

inputs0
matmul_readvariableop_resource:@@-
biasadd_readvariableop_resource:@
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:@@*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@P
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:���������@a
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:���������@w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:���������@
 
_user_specified_nameinputs
�
�
&__inference_fc_1_layer_call_fn_1031447

inputs
unknown:@@
	unknown_0:@
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *J
fERC
A__inference_fc_1_layer_call_and_return_conditional_losses_1031139o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������@`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������@: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:���������@
 
_user_specified_nameinputs
�

�
A__inference_fc_1_layer_call_and_return_conditional_losses_1031458

inputs0
matmul_readvariableop_resource:@@-
biasadd_readvariableop_resource:@
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:@@*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@P
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:���������@a
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:���������@w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:���������@
 
_user_specified_nameinputs
�	
�
)__inference_model_3_layer_call_fn_1031177
overcooked_observation
unknown:`@
	unknown_0:@
	unknown_1:@@
	unknown_2:@
	unknown_3:@
	unknown_4:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallovercooked_observationunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*(
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8� *M
fHRF
D__inference_model_3_layer_call_and_return_conditional_losses_1031162o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:���������`: : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:_ [
'
_output_shapes
:���������`
0
_user_specified_nameOvercooked_observation
�
�
)__inference_model_3_layer_call_fn_1031353

inputs
unknown:`@
	unknown_0:@
	unknown_1:@@
	unknown_2:@
	unknown_3:@
	unknown_4:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*(
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8� *M
fHRF
D__inference_model_3_layer_call_and_return_conditional_losses_1031162o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:���������`: : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:���������`
 
_user_specified_nameinputs
�

�
A__inference_fc_0_layer_call_and_return_conditional_losses_1031438

inputs0
matmul_readvariableop_resource:`@-
biasadd_readvariableop_resource:@
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:`@*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@P
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:���������@a
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:���������@w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������`: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:���������`
 
_user_specified_nameinputs
�
�
"__inference__wrapped_model_1031104
overcooked_observation=
+model_3_fc_0_matmul_readvariableop_resource:`@:
,model_3_fc_0_biasadd_readvariableop_resource:@=
+model_3_fc_1_matmul_readvariableop_resource:@@:
,model_3_fc_1_biasadd_readvariableop_resource:@?
-model_3_logits_matmul_readvariableop_resource:@<
.model_3_logits_biasadd_readvariableop_resource:
identity��#model_3/fc_0/BiasAdd/ReadVariableOp�"model_3/fc_0/MatMul/ReadVariableOp�#model_3/fc_1/BiasAdd/ReadVariableOp�"model_3/fc_1/MatMul/ReadVariableOp�%model_3/logits/BiasAdd/ReadVariableOp�$model_3/logits/MatMul/ReadVariableOp�
"model_3/fc_0/MatMul/ReadVariableOpReadVariableOp+model_3_fc_0_matmul_readvariableop_resource*
_output_shapes

:`@*
dtype0�
model_3/fc_0/MatMulMatMulovercooked_observation*model_3/fc_0/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@�
#model_3/fc_0/BiasAdd/ReadVariableOpReadVariableOp,model_3_fc_0_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0�
model_3/fc_0/BiasAddBiasAddmodel_3/fc_0/MatMul:product:0+model_3/fc_0/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@j
model_3/fc_0/ReluRelumodel_3/fc_0/BiasAdd:output:0*
T0*'
_output_shapes
:���������@�
"model_3/fc_1/MatMul/ReadVariableOpReadVariableOp+model_3_fc_1_matmul_readvariableop_resource*
_output_shapes

:@@*
dtype0�
model_3/fc_1/MatMulMatMulmodel_3/fc_0/Relu:activations:0*model_3/fc_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@�
#model_3/fc_1/BiasAdd/ReadVariableOpReadVariableOp,model_3_fc_1_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0�
model_3/fc_1/BiasAddBiasAddmodel_3/fc_1/MatMul:product:0+model_3/fc_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@j
model_3/fc_1/ReluRelumodel_3/fc_1/BiasAdd:output:0*
T0*'
_output_shapes
:���������@�
$model_3/logits/MatMul/ReadVariableOpReadVariableOp-model_3_logits_matmul_readvariableop_resource*
_output_shapes

:@*
dtype0�
model_3/logits/MatMulMatMulmodel_3/fc_1/Relu:activations:0,model_3/logits/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
%model_3/logits/BiasAdd/ReadVariableOpReadVariableOp.model_3_logits_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
model_3/logits/BiasAddBiasAddmodel_3/logits/MatMul:product:0-model_3/logits/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������n
IdentityIdentitymodel_3/logits/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp$^model_3/fc_0/BiasAdd/ReadVariableOp#^model_3/fc_0/MatMul/ReadVariableOp$^model_3/fc_1/BiasAdd/ReadVariableOp#^model_3/fc_1/MatMul/ReadVariableOp&^model_3/logits/BiasAdd/ReadVariableOp%^model_3/logits/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:���������`: : : : : : 2J
#model_3/fc_0/BiasAdd/ReadVariableOp#model_3/fc_0/BiasAdd/ReadVariableOp2H
"model_3/fc_0/MatMul/ReadVariableOp"model_3/fc_0/MatMul/ReadVariableOp2J
#model_3/fc_1/BiasAdd/ReadVariableOp#model_3/fc_1/BiasAdd/ReadVariableOp2H
"model_3/fc_1/MatMul/ReadVariableOp"model_3/fc_1/MatMul/ReadVariableOp2N
%model_3/logits/BiasAdd/ReadVariableOp%model_3/logits/BiasAdd/ReadVariableOp2L
$model_3/logits/MatMul/ReadVariableOp$model_3/logits/MatMul/ReadVariableOp:_ [
'
_output_shapes
:���������`
0
_user_specified_nameOvercooked_observation
�4
�	
 __inference__traced_save_1031572
file_prefix*
&savev2_fc_0_kernel_read_readvariableop(
$savev2_fc_0_bias_read_readvariableop*
&savev2_fc_1_kernel_read_readvariableop(
$savev2_fc_1_bias_read_readvariableop,
(savev2_logits_kernel_read_readvariableop*
&savev2_logits_bias_read_readvariableop(
$savev2_iteration_read_readvariableop	,
(savev2_learning_rate_read_readvariableop1
-savev2_adam_m_fc_0_kernel_read_readvariableop1
-savev2_adam_v_fc_0_kernel_read_readvariableop/
+savev2_adam_m_fc_0_bias_read_readvariableop/
+savev2_adam_v_fc_0_bias_read_readvariableop1
-savev2_adam_m_fc_1_kernel_read_readvariableop1
-savev2_adam_v_fc_1_kernel_read_readvariableop/
+savev2_adam_m_fc_1_bias_read_readvariableop/
+savev2_adam_v_fc_1_bias_read_readvariableop3
/savev2_adam_m_logits_kernel_read_readvariableop3
/savev2_adam_v_logits_kernel_read_readvariableop1
-savev2_adam_m_logits_bias_read_readvariableop1
-savev2_adam_v_logits_bias_read_readvariableop&
"savev2_total_1_read_readvariableop&
"savev2_count_1_read_readvariableop$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableop
savev2_const

identity_1��MergeV2Checkpointsw
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*Z
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.parta
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B
_temp/part�
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: f

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: L

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :f
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : �
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: �

SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*�

value�
B�
B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB0optimizer/_iterations/.ATTRIBUTES/VARIABLE_VALUEB3optimizer/_learning_rate/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/1/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/2/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/3/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/4/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/5/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/6/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/7/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/8/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/9/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/10/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/11/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/12/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH�
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*E
value<B:B B B B B B B B B B B B B B B B B B B B B B B B B �	
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0&savev2_fc_0_kernel_read_readvariableop$savev2_fc_0_bias_read_readvariableop&savev2_fc_1_kernel_read_readvariableop$savev2_fc_1_bias_read_readvariableop(savev2_logits_kernel_read_readvariableop&savev2_logits_bias_read_readvariableop$savev2_iteration_read_readvariableop(savev2_learning_rate_read_readvariableop-savev2_adam_m_fc_0_kernel_read_readvariableop-savev2_adam_v_fc_0_kernel_read_readvariableop+savev2_adam_m_fc_0_bias_read_readvariableop+savev2_adam_v_fc_0_bias_read_readvariableop-savev2_adam_m_fc_1_kernel_read_readvariableop-savev2_adam_v_fc_1_kernel_read_readvariableop+savev2_adam_m_fc_1_bias_read_readvariableop+savev2_adam_v_fc_1_bias_read_readvariableop/savev2_adam_m_logits_kernel_read_readvariableop/savev2_adam_v_logits_kernel_read_readvariableop-savev2_adam_m_logits_bias_read_readvariableop-savev2_adam_v_logits_bias_read_readvariableop"savev2_total_1_read_readvariableop"savev2_count_1_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableopsavev2_const"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *'
dtypes
2	�
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:�
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 f
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: Q

Identity_1IdentityIdentity:output:0^NoOp*
T0*
_output_shapes
: [
NoOpNoOp^MergeV2Checkpoints*"
_acd_function_control_output(*
_output_shapes
 "!

identity_1Identity_1:output:0*�
_input_shapes�
�: :`@:@:@@:@:@:: : :`@:`@:@:@:@@:@@:@:@:@:@::: : : : : 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:$ 

_output_shapes

:`@: 

_output_shapes
:@:$ 

_output_shapes

:@@: 

_output_shapes
:@:$ 

_output_shapes

:@: 

_output_shapes
::

_output_shapes
: :

_output_shapes
: :$	 

_output_shapes

:`@:$
 

_output_shapes

:`@: 

_output_shapes
:@: 

_output_shapes
:@:$ 

_output_shapes

:@@:$ 

_output_shapes

:@@: 

_output_shapes
:@: 

_output_shapes
:@:$ 

_output_shapes

:@:$ 

_output_shapes

:@: 

_output_shapes
:: 

_output_shapes
::

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
�
�
%__inference_signature_wrapper_1031336
overcooked_observation
unknown:`@
	unknown_0:@
	unknown_1:@@
	unknown_2:@
	unknown_3:@
	unknown_4:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallovercooked_observationunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*(
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8� *+
f&R$
"__inference__wrapped_model_1031104o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:���������`: : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:_ [
'
_output_shapes
:���������`
0
_user_specified_nameOvercooked_observation
�
�
D__inference_model_3_layer_call_and_return_conditional_losses_1031162

inputs
fc_0_1031123:`@
fc_0_1031125:@
fc_1_1031140:@@
fc_1_1031142:@ 
logits_1031156:@
logits_1031158:
identity��fc_0/StatefulPartitionedCall�fc_1/StatefulPartitionedCall�logits/StatefulPartitionedCall�
fc_0/StatefulPartitionedCallStatefulPartitionedCallinputsfc_0_1031123fc_0_1031125*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *J
fERC
A__inference_fc_0_layer_call_and_return_conditional_losses_1031122�
fc_1/StatefulPartitionedCallStatefulPartitionedCall%fc_0/StatefulPartitionedCall:output:0fc_1_1031140fc_1_1031142*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *J
fERC
A__inference_fc_1_layer_call_and_return_conditional_losses_1031139�
logits/StatefulPartitionedCallStatefulPartitionedCall%fc_1/StatefulPartitionedCall:output:0logits_1031156logits_1031158*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *L
fGRE
C__inference_logits_layer_call_and_return_conditional_losses_1031155v
IdentityIdentity'logits/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp^fc_0/StatefulPartitionedCall^fc_1/StatefulPartitionedCall^logits/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:���������`: : : : : : 2<
fc_0/StatefulPartitionedCallfc_0/StatefulPartitionedCall2<
fc_1/StatefulPartitionedCallfc_1/StatefulPartitionedCall2@
logits/StatefulPartitionedCalllogits/StatefulPartitionedCall:O K
'
_output_shapes
:���������`
 
_user_specified_nameinputs
�	
�
C__inference_logits_layer_call_and_return_conditional_losses_1031477

inputs0
matmul_readvariableop_resource:@-
biasadd_readvariableop_resource:
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:@*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������_
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:���������w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:���������@
 
_user_specified_nameinputs
�
�
D__inference_model_3_layer_call_and_return_conditional_losses_1031296
overcooked_observation
fc_0_1031280:`@
fc_0_1031282:@
fc_1_1031285:@@
fc_1_1031287:@ 
logits_1031290:@
logits_1031292:
identity��fc_0/StatefulPartitionedCall�fc_1/StatefulPartitionedCall�logits/StatefulPartitionedCall�
fc_0/StatefulPartitionedCallStatefulPartitionedCallovercooked_observationfc_0_1031280fc_0_1031282*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *J
fERC
A__inference_fc_0_layer_call_and_return_conditional_losses_1031122�
fc_1/StatefulPartitionedCallStatefulPartitionedCall%fc_0/StatefulPartitionedCall:output:0fc_1_1031285fc_1_1031287*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *J
fERC
A__inference_fc_1_layer_call_and_return_conditional_losses_1031139�
logits/StatefulPartitionedCallStatefulPartitionedCall%fc_1/StatefulPartitionedCall:output:0logits_1031290logits_1031292*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *L
fGRE
C__inference_logits_layer_call_and_return_conditional_losses_1031155v
IdentityIdentity'logits/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp^fc_0/StatefulPartitionedCall^fc_1/StatefulPartitionedCall^logits/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:���������`: : : : : : 2<
fc_0/StatefulPartitionedCallfc_0/StatefulPartitionedCall2<
fc_1/StatefulPartitionedCallfc_1/StatefulPartitionedCall2@
logits/StatefulPartitionedCalllogits/StatefulPartitionedCall:_ [
'
_output_shapes
:���������`
0
_user_specified_nameOvercooked_observation
�e
�
#__inference__traced_restore_1031654
file_prefix.
assignvariableop_fc_0_kernel:`@*
assignvariableop_1_fc_0_bias:@0
assignvariableop_2_fc_1_kernel:@@*
assignvariableop_3_fc_1_bias:@2
 assignvariableop_4_logits_kernel:@,
assignvariableop_5_logits_bias:&
assignvariableop_6_iteration:	 *
 assignvariableop_7_learning_rate: 7
%assignvariableop_8_adam_m_fc_0_kernel:`@7
%assignvariableop_9_adam_v_fc_0_kernel:`@2
$assignvariableop_10_adam_m_fc_0_bias:@2
$assignvariableop_11_adam_v_fc_0_bias:@8
&assignvariableop_12_adam_m_fc_1_kernel:@@8
&assignvariableop_13_adam_v_fc_1_kernel:@@2
$assignvariableop_14_adam_m_fc_1_bias:@2
$assignvariableop_15_adam_v_fc_1_bias:@:
(assignvariableop_16_adam_m_logits_kernel:@:
(assignvariableop_17_adam_v_logits_kernel:@4
&assignvariableop_18_adam_m_logits_bias:4
&assignvariableop_19_adam_v_logits_bias:%
assignvariableop_20_total_1: %
assignvariableop_21_count_1: #
assignvariableop_22_total: #
assignvariableop_23_count: 
identity_25��AssignVariableOp�AssignVariableOp_1�AssignVariableOp_10�AssignVariableOp_11�AssignVariableOp_12�AssignVariableOp_13�AssignVariableOp_14�AssignVariableOp_15�AssignVariableOp_16�AssignVariableOp_17�AssignVariableOp_18�AssignVariableOp_19�AssignVariableOp_2�AssignVariableOp_20�AssignVariableOp_21�AssignVariableOp_22�AssignVariableOp_23�AssignVariableOp_3�AssignVariableOp_4�AssignVariableOp_5�AssignVariableOp_6�AssignVariableOp_7�AssignVariableOp_8�AssignVariableOp_9�

RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*�

value�
B�
B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB0optimizer/_iterations/.ATTRIBUTES/VARIABLE_VALUEB3optimizer/_learning_rate/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/1/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/2/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/3/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/4/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/5/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/6/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/7/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/8/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/9/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/10/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/11/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/12/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH�
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*E
value<B:B B B B B B B B B B B B B B B B B B B B B B B B B �
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*x
_output_shapesf
d:::::::::::::::::::::::::*'
dtypes
2	[
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOpAssignVariableOpassignvariableop_fc_0_kernelIdentity:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_1AssignVariableOpassignvariableop_1_fc_0_biasIdentity_1:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_2AssignVariableOpassignvariableop_2_fc_1_kernelIdentity_2:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_3AssignVariableOpassignvariableop_3_fc_1_biasIdentity_3:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_4AssignVariableOp assignvariableop_4_logits_kernelIdentity_4:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_5AssignVariableOpassignvariableop_5_logits_biasIdentity_5:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0	*
_output_shapes
:�
AssignVariableOp_6AssignVariableOpassignvariableop_6_iterationIdentity_6:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0	]

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_7AssignVariableOp assignvariableop_7_learning_rateIdentity_7:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_8AssignVariableOp%assignvariableop_8_adam_m_fc_0_kernelIdentity_8:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_9AssignVariableOp%assignvariableop_9_adam_v_fc_0_kernelIdentity_9:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_10AssignVariableOp$assignvariableop_10_adam_m_fc_0_biasIdentity_10:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_11AssignVariableOp$assignvariableop_11_adam_v_fc_0_biasIdentity_11:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_12AssignVariableOp&assignvariableop_12_adam_m_fc_1_kernelIdentity_12:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_13AssignVariableOp&assignvariableop_13_adam_v_fc_1_kernelIdentity_13:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_14AssignVariableOp$assignvariableop_14_adam_m_fc_1_biasIdentity_14:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_15AssignVariableOp$assignvariableop_15_adam_v_fc_1_biasIdentity_15:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_16AssignVariableOp(assignvariableop_16_adam_m_logits_kernelIdentity_16:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_17AssignVariableOp(assignvariableop_17_adam_v_logits_kernelIdentity_17:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_18AssignVariableOp&assignvariableop_18_adam_m_logits_biasIdentity_18:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_19AssignVariableOp&assignvariableop_19_adam_v_logits_biasIdentity_19:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_20AssignVariableOpassignvariableop_20_total_1Identity_20:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_21AssignVariableOpassignvariableop_21_count_1Identity_21:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_22AssignVariableOpassignvariableop_22_totalIdentity_22:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_23AssignVariableOpassignvariableop_23_countIdentity_23:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0Y
NoOpNoOp"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 �
Identity_24Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: W
Identity_25IdentityIdentity_24:output:0^NoOp_1*
T0*
_output_shapes
: �
NoOp_1NoOp^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*"
_acd_function_control_output(*
_output_shapes
 "#
identity_25Identity_25:output:0*E
_input_shapes4
2: : : : : : : : : : : : : : : : : : : : : : : : : 2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12*
AssignVariableOp_10AssignVariableOp_102*
AssignVariableOp_11AssignVariableOp_112*
AssignVariableOp_12AssignVariableOp_122*
AssignVariableOp_13AssignVariableOp_132*
AssignVariableOp_14AssignVariableOp_142*
AssignVariableOp_15AssignVariableOp_152*
AssignVariableOp_16AssignVariableOp_162*
AssignVariableOp_17AssignVariableOp_172*
AssignVariableOp_18AssignVariableOp_182*
AssignVariableOp_19AssignVariableOp_192(
AssignVariableOp_2AssignVariableOp_22*
AssignVariableOp_20AssignVariableOp_202*
AssignVariableOp_21AssignVariableOp_212*
AssignVariableOp_22AssignVariableOp_222*
AssignVariableOp_23AssignVariableOp_232(
AssignVariableOp_3AssignVariableOp_32(
AssignVariableOp_4AssignVariableOp_42(
AssignVariableOp_5AssignVariableOp_52(
AssignVariableOp_6AssignVariableOp_62(
AssignVariableOp_7AssignVariableOp_72(
AssignVariableOp_8AssignVariableOp_82(
AssignVariableOp_9AssignVariableOp_9:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix"�
L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*�
serving_default�
Y
Overcooked_observation?
(serving_default_Overcooked_observation:0���������`:
logits0
StatefulPartitionedCall:0���������tensorflow/serving/predict:�r
�
layer-0
layer_with_weights-0
layer-1
layer_with_weights-1
layer-2
layer_with_weights-2
layer-3
	variables
trainable_variables
regularization_losses
	keras_api
	__call__
*
&call_and_return_all_conditional_losses
_default_save_signature
	optimizer

signatures"
_tf_keras_network
"
_tf_keras_input_layer
�
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses

kernel
bias"
_tf_keras_layer
�
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses

kernel
bias"
_tf_keras_layer
�
	variables
trainable_variables
 regularization_losses
!	keras_api
"__call__
*#&call_and_return_all_conditional_losses

$kernel
%bias"
_tf_keras_layer
J
0
1
2
3
$4
%5"
trackable_list_wrapper
J
0
1
2
3
$4
%5"
trackable_list_wrapper
 "
trackable_list_wrapper
�
&non_trainable_variables

'layers
(metrics
)layer_regularization_losses
*layer_metrics
	variables
trainable_variables
regularization_losses
	__call__
_default_save_signature
*
&call_and_return_all_conditional_losses
&
"call_and_return_conditional_losses"
_generic_user_object
�
+trace_0
,trace_1
-trace_2
.trace_32�
)__inference_model_3_layer_call_fn_1031177
)__inference_model_3_layer_call_fn_1031353
)__inference_model_3_layer_call_fn_1031370
)__inference_model_3_layer_call_fn_1031277�
���
FullArgSpec1
args)�&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z+trace_0z,trace_1z-trace_2z.trace_3
�
/trace_0
0trace_1
1trace_2
2trace_32�
D__inference_model_3_layer_call_and_return_conditional_losses_1031394
D__inference_model_3_layer_call_and_return_conditional_losses_1031418
D__inference_model_3_layer_call_and_return_conditional_losses_1031296
D__inference_model_3_layer_call_and_return_conditional_losses_1031315�
���
FullArgSpec1
args)�&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z/trace_0z0trace_1z1trace_2z2trace_3
�B�
"__inference__wrapped_model_1031104Overcooked_observation"�
���
FullArgSpec
args� 
varargsjargs
varkwjkwargs
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�
3
_variables
4_iterations
5_learning_rate
6_index_dict
7
_momentums
8_velocities
9_update_step_xla"
experimentalOptimizer
,
:serving_default"
signature_map
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
;non_trainable_variables

<layers
=metrics
>layer_regularization_losses
?layer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
�
@trace_02�
&__inference_fc_0_layer_call_fn_1031427�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z@trace_0
�
Atrace_02�
A__inference_fc_0_layer_call_and_return_conditional_losses_1031438�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 zAtrace_0
:`@2fc_0/kernel
:@2	fc_0/bias
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
Bnon_trainable_variables

Clayers
Dmetrics
Elayer_regularization_losses
Flayer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
�
Gtrace_02�
&__inference_fc_1_layer_call_fn_1031447�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 zGtrace_0
�
Htrace_02�
A__inference_fc_1_layer_call_and_return_conditional_losses_1031458�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 zHtrace_0
:@@2fc_1/kernel
:@2	fc_1/bias
.
$0
%1"
trackable_list_wrapper
.
$0
%1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
Inon_trainable_variables

Jlayers
Kmetrics
Llayer_regularization_losses
Mlayer_metrics
	variables
trainable_variables
 regularization_losses
"__call__
*#&call_and_return_all_conditional_losses
&#"call_and_return_conditional_losses"
_generic_user_object
�
Ntrace_02�
(__inference_logits_layer_call_fn_1031467�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 zNtrace_0
�
Otrace_02�
C__inference_logits_layer_call_and_return_conditional_losses_1031477�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 zOtrace_0
:@2logits/kernel
:2logits/bias
 "
trackable_list_wrapper
<
0
1
2
3"
trackable_list_wrapper
.
P0
Q1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
)__inference_model_3_layer_call_fn_1031177Overcooked_observation"�
���
FullArgSpec1
args)�&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
)__inference_model_3_layer_call_fn_1031353inputs"�
���
FullArgSpec1
args)�&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
)__inference_model_3_layer_call_fn_1031370inputs"�
���
FullArgSpec1
args)�&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
)__inference_model_3_layer_call_fn_1031277Overcooked_observation"�
���
FullArgSpec1
args)�&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
D__inference_model_3_layer_call_and_return_conditional_losses_1031394inputs"�
���
FullArgSpec1
args)�&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
D__inference_model_3_layer_call_and_return_conditional_losses_1031418inputs"�
���
FullArgSpec1
args)�&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
D__inference_model_3_layer_call_and_return_conditional_losses_1031296Overcooked_observation"�
���
FullArgSpec1
args)�&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
D__inference_model_3_layer_call_and_return_conditional_losses_1031315Overcooked_observation"�
���
FullArgSpec1
args)�&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
~
40
R1
S2
T3
U4
V5
W6
X7
Y8
Z9
[10
\11
]12"
trackable_list_wrapper
:	 2	iteration
: 2learning_rate
 "
trackable_dict_wrapper
J
R0
T1
V2
X3
Z4
\5"
trackable_list_wrapper
J
S0
U1
W2
Y3
[4
]5"
trackable_list_wrapper
�2��
���
FullArgSpec2
args*�'
jself

jgradient

jvariable
jkey
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 0
�B�
%__inference_signature_wrapper_1031336Overcooked_observation"�
���
FullArgSpec
args� 
varargs
 
varkwjkwargs
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
&__inference_fc_0_layer_call_fn_1031427inputs"�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
A__inference_fc_0_layer_call_and_return_conditional_losses_1031438inputs"�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
&__inference_fc_1_layer_call_fn_1031447inputs"�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
A__inference_fc_1_layer_call_and_return_conditional_losses_1031458inputs"�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
(__inference_logits_layer_call_fn_1031467inputs"�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
C__inference_logits_layer_call_and_return_conditional_losses_1031477inputs"�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
N
^	variables
_	keras_api
	`total
	acount"
_tf_keras_metric
^
b	variables
c	keras_api
	dtotal
	ecount
f
_fn_kwargs"
_tf_keras_metric
": `@2Adam/m/fc_0/kernel
": `@2Adam/v/fc_0/kernel
:@2Adam/m/fc_0/bias
:@2Adam/v/fc_0/bias
": @@2Adam/m/fc_1/kernel
": @@2Adam/v/fc_1/kernel
:@2Adam/m/fc_1/bias
:@2Adam/v/fc_1/bias
$:"@2Adam/m/logits/kernel
$:"@2Adam/v/logits/kernel
:2Adam/m/logits/bias
:2Adam/v/logits/bias
.
`0
a1"
trackable_list_wrapper
-
^	variables"
_generic_user_object
:  (2total
:  (2count
.
d0
e1"
trackable_list_wrapper
-
b	variables"
_generic_user_object
:  (2total
:  (2count
 "
trackable_dict_wrapper�
"__inference__wrapped_model_1031104z$%?�<
5�2
0�-
Overcooked_observation���������`
� "/�,
*
logits �
logits����������
A__inference_fc_0_layer_call_and_return_conditional_losses_1031438c/�,
%�"
 �
inputs���������`
� ",�)
"�
tensor_0���������@
� �
&__inference_fc_0_layer_call_fn_1031427X/�,
%�"
 �
inputs���������`
� "!�
unknown���������@�
A__inference_fc_1_layer_call_and_return_conditional_losses_1031458c/�,
%�"
 �
inputs���������@
� ",�)
"�
tensor_0���������@
� �
&__inference_fc_1_layer_call_fn_1031447X/�,
%�"
 �
inputs���������@
� "!�
unknown���������@�
C__inference_logits_layer_call_and_return_conditional_losses_1031477c$%/�,
%�"
 �
inputs���������@
� ",�)
"�
tensor_0���������
� �
(__inference_logits_layer_call_fn_1031467X$%/�,
%�"
 �
inputs���������@
� "!�
unknown����������
D__inference_model_3_layer_call_and_return_conditional_losses_1031296$%G�D
=�:
0�-
Overcooked_observation���������`
p 

 
� ",�)
"�
tensor_0���������
� �
D__inference_model_3_layer_call_and_return_conditional_losses_1031315$%G�D
=�:
0�-
Overcooked_observation���������`
p

 
� ",�)
"�
tensor_0���������
� �
D__inference_model_3_layer_call_and_return_conditional_losses_1031394o$%7�4
-�*
 �
inputs���������`
p 

 
� ",�)
"�
tensor_0���������
� �
D__inference_model_3_layer_call_and_return_conditional_losses_1031418o$%7�4
-�*
 �
inputs���������`
p

 
� ",�)
"�
tensor_0���������
� �
)__inference_model_3_layer_call_fn_1031177t$%G�D
=�:
0�-
Overcooked_observation���������`
p 

 
� "!�
unknown����������
)__inference_model_3_layer_call_fn_1031277t$%G�D
=�:
0�-
Overcooked_observation���������`
p

 
� "!�
unknown����������
)__inference_model_3_layer_call_fn_1031353d$%7�4
-�*
 �
inputs���������`
p 

 
� "!�
unknown����������
)__inference_model_3_layer_call_fn_1031370d$%7�4
-�*
 �
inputs���������`
p

 
� "!�
unknown����������
%__inference_signature_wrapper_1031336�$%Y�V
� 
O�L
J
Overcooked_observation0�-
overcooked_observation���������`"/�,
*
logits �
logits���������