"?9
BHostIDLE"IDLE1    @??@A    @??@a???????i????????Unknown
sHost_FusedMatMul"sequential_2/dense_4/Relu(1     H?@9     H?@A     H?@I     H?@a????凚?i_? T????Unknown
sHostDataset"Iterator::Model::ParallelMapV2(1     ?x@9     ?x@A     ?x@I     ?x@a???|????i? E???Unknown
^HostGatherV2"GatherV2(1     Pt@9     Pt@A     Pt@I     Pt@a???V?z??i[1??????Unknown
}HostMatMul")gradient_tape/sequential_2/dense_4/MatMul(1     @c@9     @c@A     @c@I     @c@a53??{?i?Ll??????Unknown
uHostFlushSummaryWriter"FlushSummaryWriter(1     @X@9     @X@A     @X@I     @X@a<??
??q?i??????Unknown?
?HostDataset"/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap(1     ?T@9     ?T@A     @R@I     @R@a?C?lI|j?i]??Ex0???Unknown
?HostResourceApplyGradientDescent"+SGD/SGD/update/ResourceApplyGradientDescent(1      R@9      R@A      R@I      R@a%??gj?iA?魗J???Unknown
}	HostMatMul")gradient_tape/sequential_2/dense_5/MatMul(1     ?B@9     ?B@A     ?B@I     ?B@a^?m?*?Z?i?XCX???Unknown
v
Host_FusedMatMul"sequential_2/dense_5/BiasAdd(1     ?@@9     ?@@A     ?@@I     ?@@aw??P?W?ifYS?c???Unknown
?Host#SparseSoftmaxCrossEntropyWithLogits"gsparse_categorical_crossentropy/SparseSoftmaxCrossEntropyWithLogits/SparseSoftmaxCrossEntropyWithLogits(1     ?@@9     ?@@A     ?@@I     ?@@aw??P?W?i???b?o???Unknown
?HostMul"Ugradient_tape/sparse_categorical_crossentropy/SparseSoftmaxCrossEntropyWithLogits/mul(1      =@9      =@A      =@I      =@a???U?i?F??{z???Unknown
HostMatMul"+gradient_tape/sequential_2/dense_5/MatMul_1(1      <@9      <@A      <@I      <@aU???PQT?i???????Unknown
?HostDataset"5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat(1      ?@9      ?@A      :@I      :@a?k???R?i???z????Unknown
dHostDataset"Iterator::Model(1      z@9      z@A      4@I      4@aᭈsM?i1?iU????Unknown
qHostSoftmax"sequential_2/dense_5/Softmax(1      4@9      4@A      4@I      4@aᭈsM?i??K??????Unknown
?HostResourceApplyGradientDescent"-SGD/SGD/update_3/ResourceApplyGradientDescent(1      3@9      3@A      3@I      3@a?b????K?i?U?o{????Unknown
iHostWriteSummary"WriteSummary(1      3@9      3@A      3@I      3@a?b????K?i??,+`????Unknown?
?HostReadVariableOp"+sequential_2/dense_4/BiasAdd/ReadVariableOp(1      1@9      1@A      1@I      1@a?e-4??H?i4׹#?????Unknown
lHostIteratorGetNext"IteratorGetNext(1      *@9      *@A      *@I      *@a?k???B?i??B????Unknown
?HostBiasAddGrad"6gradient_tape/sequential_2/dense_5/BiasAdd/BiasAddGrad(1      *@9      *@A      *@I      *@a?k???B?i?BF	?????Unknown
?HostResourceApplyGradientDescent"-SGD/SGD/update_2/ResourceApplyGradientDescent(1      (@9      (@A      (@I      (@an?REjA?ieÚ?T????Unknown
eHost
LogicalAnd"
LogicalAnd(1      &@9      &@A      &@I      &@a??X???i??}JR????Unknown?
vHostAssignAddVariableOp"AssignAddVariableOp_2(1      $@9      $@A      $@I      $@aᭈs=?i=???????Unknown
?HostDataset"?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::TensorSlice(1      $@9      $@A      $@I      $@aᭈs=?i??_??????Unknown
?HostResourceApplyGradientDescent"-SGD/SGD/update_1/ResourceApplyGradientDescent(1      "@9      "@A      "@I      "@a%??g:?iV_??????Unknown
?HostSum"1sparse_categorical_crossentropy/weighted_loss/Sum(1      "@9      "@A      "@I      "@a%??g:?i?z^?????Unknown
xHostDataset"#Iterator::Model::ParallelMapV2::Zip(1     ?^@9     ?^@A       @I       @a=?Wm\87?i?%??????Unknown
uHostReadVariableOp"div_no_nan/ReadVariableOp(1       @9       @A       @I       @a=?Wm\87?i??y??????Unknown
?HostReadVariableOp"*sequential_2/dense_4/MatMul/ReadVariableOp(1       @9       @A       @I       @a=?Wm\87?i?{??????Unknown
ZHostArgMax"ArgMax(1      @9      @A      @I      @aU???PQ4?iGq#[????Unknown
` HostGatherV2"
GatherV2_1(1      @9      @A      @I      @aU???PQ4?i?f?8?????Unknown
V!HostSum"Sum_2(1      @9      @A      @I      @aU???PQ4?i?\[bo????Unknown
?"HostBiasAddGrad"6gradient_tape/sequential_2/dense_4/BiasAdd/BiasAddGrad(1      @9      @A      @I      @aU???PQ4?iRw??????Unknown
v#HostCast"$sparse_categorical_crossentropy/Cast(1      @9      @A      @I      @aU???PQ4?i?G???????Unknown
?$HostDataset"AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor(1      @9      @A      @I      @aᭈs-?i???T????Unknown
s%HostReadVariableOp"SGD/Cast/ReadVariableOp(1      @9      @A      @I      @aᭈs-?iw]?$????Unknown
u&HostReadVariableOp"SGD/Cast_1/ReadVariableOp(1      @9      @A      @I      @aᭈs-?iU?<??????Unknown
?'HostReluGrad"+gradient_tape/sequential_2/dense_4/ReluGrad(1      @9      @A      @I      @aᭈs-?i3suS?????Unknown
v(HostAssignAddVariableOp"AssignAddVariableOp_4(1      @9      @A      @I      @a=?Wm\8'?i?H<?8????Unknown
|)HostAssignAddVariableOp"SGD/SGD/AssignAddVariableOp(1      @9      @A      @I      @a=?Wm\8'?i/_?????Unknown
`*HostDivNoNan"
div_no_nan(1      @9      @A      @I      @a=?Wm\8'?i????????Unknown
b+HostDivNoNan"div_no_nan_1(1      @9      @A      @I      @a=?Wm\8'?i+ɐj?????Unknown
t,HostAssignAddVariableOp"AssignAddVariableOp(1      @9      @A      @I      @an?REj!?iJ???????Unknown
v-HostAssignAddVariableOp"AssignAddVariableOp_3(1      @9      @A      @I      @an?REj!?ii	;??????Unknown
X.HostCast"Cast_2(1      @9      @A      @I      @an?REj!?i?)?W?????Unknown
X/HostCast"Cast_3(1      @9      @A      @I      @an?REj!?i?I???????Unknown
X0HostEqual"Equal(1      @9      @A      @I      @an?REj!?i?i:?????Unknown
?1HostReadVariableOp"+sequential_2/dense_5/BiasAdd/ReadVariableOp(1      @9      @A      @I      @an?REj!?i剏D????Unknown
?2HostDivNoNan"3sparse_categorical_crossentropy/weighted_loss/value(1      @9      @A      @I      @an?REj!?i???1????Unknown
v3HostAssignAddVariableOp"AssignAddVariableOp_1(1       @9       @A       @I       @a=?Wm\8?i?ȫ?????Unknown
T4HostMul"Mul(1       @9       @A       @I       @a=?Wm\8?i??n?????Unknown
w5HostReadVariableOp"div_no_nan/ReadVariableOp_1(1       @9       @A       @I       @a=?Wm\8?iA??1_????Unknown
y6HostReadVariableOp"div_no_nan_1/ReadVariableOp_1(1       @9       @A       @I       @a=?Wm\8?i Ur?????Unknown
?7HostReadVariableOp"*sequential_2/dense_5/MatMul/ReadVariableOp(1       @9       @A       @I       @a=?Wm\8?i??U??????Unknown
?8HostCast"`sparse_categorical_crossentropy/ArithmeticOptimizer/ReorderCastLikeAndValuePreserving_float_Cast(1       @9       @A       @I       @a=?Wm\8?i~*9z?????Unknown
?9HostCast"bsparse_categorical_crossentropy/ArithmeticOptimizer/ReorderCastLikeAndValuePreserving_int64_Cast_1(1       @9       @A       @I       @a=?Wm\8?i=?=F????Unknown
a:HostIdentity"Identity(1      ??9      ??A      ??I      ??a=?Wm\8?i?J??????Unknown?
w;HostReadVariableOp"div_no_nan_1/ReadVariableOp(1      ??9      ??A      ??I      ??a=?Wm\8?i?????????Unknown*?9
sHost_FusedMatMul"sequential_2/dense_4/Relu(1     H?@9     H?@A     H?@I     H?@a?)|?? ??i?)|?? ???Unknown
sHostDataset"Iterator::Model::ParallelMapV2(1     ?x@9     ?x@A     ?x@I     ?x@a?$e???iV?.?#????Unknown
^HostGatherV2"GatherV2(1     Pt@9     Pt@A     Pt@I     Pt@a+?¢T???ivH??????Unknown
}HostMatMul")gradient_tape/sequential_2/dense_4/MatMul(1     @c@9     @c@A     @c@I     @c@a???rZ???i???@?7???Unknown
uHostFlushSummaryWriter"FlushSummaryWriter(1     @X@9     @X@A     @X@I     @X@aj???Ս??i)???o????Unknown?
?HostDataset"/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap(1     ?T@9     ?T@A     @R@I     @R@a??O?A???i5?b?????Unknown
?HostResourceApplyGradientDescent"+SGD/SGD/update/ResourceApplyGradientDescent(1      R@9      R@A      R@I      R@a?w죻???i?]?r߻???Unknown
}HostMatMul")gradient_tape/sequential_2/dense_5/MatMul(1     ?B@9     ?B@A     ?B@I     ?B@a?	???4??i?V??E???Unknown
v	Host_FusedMatMul"sequential_2/dense_5/BiasAdd(1     ?@@9     ?@@A     ?@@I     ?@@a??1W-???i??gJ????Unknown
?
Host#SparseSoftmaxCrossEntropyWithLogits"gsparse_categorical_crossentropy/SparseSoftmaxCrossEntropyWithLogits/SparseSoftmaxCrossEntropyWithLogits(1     ?@@9     ?@@A     ?@@I     ?@@a??1W-???i6?;???Unknown
?HostMul"Ugradient_tape/sparse_categorical_crossentropy/SparseSoftmaxCrossEntropyWithLogits/mul(1      =@9      =@A      =@I      =@a0???????ivH?????Unknown
HostMatMul"+gradient_tape/sequential_2/dense_5/MatMul_1(1      <@9      <@A      <@I      <@a?o?
??i{5????Unknown
?HostDataset"5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat(1      ?@9      ?@A      :@I      :@aېUЀ.??i????o???Unknown
dHostDataset"Iterator::Model(1      z@9      z@A      4@I      4@a3??왂?i???>????Unknown
qHostSoftmax"sequential_2/dense_5/Softmax(1      4@9      4@A      4@I      4@a3??왂?i??;{????Unknown
?HostResourceApplyGradientDescent"-SGD/SGD/update_3/ResourceApplyGradientDescent(1      3@9      3@A      3@I      3@a?yIԫ??i?a?UK???Unknown
iHostWriteSummary"WriteSummary(1      3@9      3@A      3@I      3@a?yIԫ??i???????Unknown?
?HostReadVariableOp"+sequential_2/dense_4/BiasAdd/ReadVariableOp(1      1@9      1@A      1@I      1@a????E??i???C????Unknown
lHostIteratorGetNext"IteratorGetNext(1      *@9      *@A      *@I      *@aېUЀ.x?i??!?????Unknown
?HostBiasAddGrad"6gradient_tape/sequential_2/dense_5/BiasAdd/BiasAddGrad(1      *@9      *@A      *@I      *@aېUЀ.x?ib¬?1???Unknown
?HostResourceApplyGradientDescent"-SGD/SGD/update_2/ResourceApplyGradientDescent(1      (@9      (@A      (@I      (@a?J;?ORv?i???K?^???Unknown
eHost
LogicalAnd"
LogicalAnd(1      &@9      &@A      &@I      &@ak!:vt?i?A??????Unknown?
vHostAssignAddVariableOp"AssignAddVariableOp_2(1      $@9      $@A      $@I      $@a3????r?i9(b¬???Unknown
?HostDataset"?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::TensorSlice(1      $@9      $@A      $@I      $@a3????r?i?5?;?????Unknown
?HostResourceApplyGradientDescent"-SGD/SGD/update_1/ResourceApplyGradientDescent(1      "@9      "@A      "@I      "@a?w죻?p?i?E?q????Unknown
?HostSum"1sparse_categorical_crossentropy/weighted_loss/Sum(1      "@9      "@A      "@I      "@a?w죻?p?i???*????Unknown
xHostDataset"#Iterator::Model::ParallelMapV2::Zip(1     ?^@9     ?^@A       @I       @a?c???m?i??>??2???Unknown
uHostReadVariableOp"div_no_nan/ReadVariableOp(1       @9       @A       @I       @a?c???m?i]0?SsP???Unknown
?HostReadVariableOp"*sequential_2/dense_4/MatMul/ReadVariableOp(1       @9       @A       @I       @a?c???m?i?ԡh6n???Unknown
ZHostArgMax"ArgMax(1      @9      @A      @I      @a?o?
j?i?D?A????Unknown
`HostGatherV2"
GatherV2_1(1      @9      @A      @I      @a?o?
j?io???K????Unknown
V HostSum"Sum_2(1      @9      @A      @I      @a?o?
j?iF$?~V????Unknown
?!HostBiasAddGrad"6gradient_tape/sequential_2/dense_4/BiasAdd/BiasAddGrad(1      @9      @A      @I      @a?o?
j?i?1a????Unknown
v"HostCast"$sparse_categorical_crossentropy/Cast(1      @9      @A      @I      @a?o?
j?i?+?k????Unknown
?#HostDataset"AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor(1      @9      @A      @I      @a3????b?i?
????Unknown
s$HostReadVariableOp"SGD/Cast/ReadVariableOp(1      @9      @A      @I      @a3????b?ip	?????Unknown
u%HostReadVariableOp"SGD/Cast_1/ReadVariableOp(1      @9      @A      @I      @a3????b?i.??9(???Unknown
?&HostReluGrad"+gradient_tape/sequential_2/dense_4/ReluGrad(1      @9      @A      @I      @a3????b?i????:???Unknown
v'HostAssignAddVariableOp"AssignAddVariableOp_4(1      @9      @A      @I      @a?c???]?i??!?I???Unknown
|(HostAssignAddVariableOp"SGD/SGD/AssignAddVariableOp(1      @9      @A      @I      @a?c???]?iPØ??X???Unknown
`)HostDivNoNan"
div_no_nan(1      @9      @A      @I      @a?c???]?i???5xg???Unknown
b*HostDivNoNan"div_no_nan_1(1      @9      @A      @I      @a?c???]?i?gJ?Yv???Unknown
t+HostAssignAddVariableOp"AssignAddVariableOp(1      @9      @A      @I      @a?J;?ORV?iY肁???Unknown
v,HostAssignAddVariableOp"AssignAddVariableOp_3(1      @9      @A      @I      @a?J;?ORV?i????????Unknown
X-HostCast"Cast_2(1      @9      @A      @I      @a?J;?ORV?i?@?7՗???Unknown
X.HostCast"Cast_3(1      @9      @A      @I      @a?J;?ORV?iH?T_?????Unknown
X/HostEqual"Equal(1      @9      @A      @I      @a?J;?ORV?i?{?'????Unknown
?0HostReadVariableOp"+sequential_2/dense_5/BiasAdd/ReadVariableOp(1      @9      @A      @I      @a?J;?ORV?i?ڮP????Unknown
?1HostDivNoNan"3sparse_categorical_crossentropy/weighted_loss/value(1      @9      @A      @I      @a?J;?ORV?i7???y????Unknown
v2HostAssignAddVariableOp"AssignAddVariableOp_1(1       @9       @A       @I       @a?c???M?iP ɛ?????Unknown
T3HostMul"Mul(1       @9       @A       @I       @a?c???M?ii??`[????Unknown
w4HostReadVariableOp"div_no_nan/ReadVariableOp_1(1       @9       @A       @I       @a?c???M?i??!&?????Unknown
y5HostReadVariableOp"div_no_nan_1/ReadVariableOp_1(1       @9       @A       @I       @a?c???M?i?[N?<????Unknown
?6HostReadVariableOp"*sequential_2/dense_5/MatMul/ReadVariableOp(1       @9       @A       @I       @a?c???M?i??z??????Unknown
?7HostCast"`sparse_categorical_crossentropy/ArithmeticOptimizer/ReorderCastLikeAndValuePreserving_float_Cast(1       @9       @A       @I       @a?c???M?i?-?u????Unknown
?8HostCast"bsparse_categorical_crossentropy/ArithmeticOptimizer/ReorderCastLikeAndValuePreserving_int64_Cast_1(1       @9       @A       @I       @a?c???M?i???:?????Unknown
a9HostIdentity"Identity(1      ??9      ??A      ??I      ??a?c???=?ir?i?G????Unknown?
w:HostReadVariableOp"div_no_nan_1/ReadVariableOp(1      ??9      ??A      ??I      ??a?c???=?i?????????Unknown2CPU