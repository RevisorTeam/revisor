import tensorrt as trt
TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

model_name = 'M'
MAX_BATCH_SIZE = 40
# 0x384x288x3 - EvoPose2d M model
net_w, net_h = 384, 288			# M
# net_w, net_h = 256, 192		# S
# net_w, net_h = 512, 384		# L

onnx_model_name = 'evopose2d_{}'.format(model_name)
onnx_model_path = 'models/{}.onnx'.format(onnx_model_name)
trt_path = 'models/{}_batch{}.trt'.format(onnx_model_name, MAX_BATCH_SIZE)

EXPLICIT_BATCH = 1 << (int)(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
with trt.Builder(TRT_LOGGER) as builder, builder.create_network(
		EXPLICIT_BATCH) as network, trt.OnnxParser(network, TRT_LOGGER) as parser:
	with open(onnx_model_path, 'rb') as model:
		if not parser.parse(model.read()):
			for error in range(parser.num_errors):
				print(parser.get_error(error))

	builder.max_batch_size = MAX_BATCH_SIZE

	config = builder.create_builder_config()
	config.max_workspace_size = 1 << 30
	config.set_flag(trt.BuilderFlag.GPU_FALLBACK)
	config.set_flag(trt.BuilderFlag.FP16)

	profile = builder.create_optimization_profile()

	profile.set_shape(

		# Input tensor name:
		# 'input_1:0',  	# uncomment if you have NVIDIA 20xx
		'input_1',			# uncomment if you have NVIDIA 30xx

		(MAX_BATCH_SIZE, net_w, net_h, 3),  	# min shape
		(MAX_BATCH_SIZE, net_w, net_h, 3),  	# opt shape
		(MAX_BATCH_SIZE, net_w, net_h, 3)  		# max shape
	)

	config.add_optimization_profile(profile)

	engine = builder.build_engine(network, config)

	# Save serialized trt engine to file
	serialized_engine = engine.serialize()
	with open(trt_path, 'wb') as f:
		f.write(engine.serialize())
