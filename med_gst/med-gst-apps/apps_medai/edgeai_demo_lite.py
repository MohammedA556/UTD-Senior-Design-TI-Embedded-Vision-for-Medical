import yaml
import gst_wrapper
import config_parser
from edgeai_dl_inferer import ModelConfig
from infer_pipe import InferPipe

class EdgeAIDemoLite:
    def __init__(self, config):
        self.config = config
        self.models = {}
        self.inputs = {}
        self.outputs = {}
        self.flows = []
        self.infer_pipes = []

        # Parse inputs/models/outputs exactly like TI does,
        # but without curses, utils.args, or mosaic logic.
        for f in config["flows"]:
            flow = config["flows"][f]
            input_name, model_name, output_name = flow[:3]

            # Input
            if input_name not in self.inputs:
                inp_cfg = config["inputs"][input_name]
                self.inputs[input_name] = config_parser.Input(inp_cfg)

            # Model
            if model_name not in self.models:
                mdl_cfg = config["models"][model_name]
                mdl = ModelConfig(mdl_cfg["model_path"], enable_tidl=False, core_id=1)
                mdl.create_runtime()
                self.models[model_name] = mdl

            # Output
            if output_name not in self.outputs:
                out_cfg = config["outputs"][output_name]
                self.outputs[output_name] = config_parser.Output(out_cfg, config["title"])

        # Build flows
        for input_name in self.inputs:
            input_obj = self.inputs[input_name]
            subflows = []

            for f in config["flows"]:
                flow = config["flows"][f]
                if flow[0] != input_name:
                    continue

                model_name = flow[1]
                output_name = flow[2]

                model_obj = self.models[model_name]
                output_obj = self.outputs[output_name]

                subflows.append([model_obj, [output_obj], [None]])

            self.flows.append(config_parser.Flow(input_obj, subflows, None))

        # Build GStreamer pipeline
        self.src_pipes, self.sink_pipe = gst_wrapper.get_gst_pipe(self.flows, self.outputs)
        self.gst_pipe = gst_wrapper.GstPipe(self.src_pipes, self.sink_pipe)

        # Attach pipe to outputs
        for o in self.outputs.values():
            o.gst_pipe = self.gst_pipe

        # Build infer pipes
        for f in self.flows:
            for s in f.sub_flows:
                self.infer_pipes.append(InferPipe(s, self.gst_pipe))

    def start(self):
        self.gst_pipe.start()
        for p in self.infer_pipes:
            p.start()

    def stop(self):
        for p in self.infer_pipes:
            p.stop()
        self.gst_pipe.free()

    def get_frame(self):
        # The output object exposes a buffer getter
        # Usually something like: output_obj.get_buffer()
        # Find the first output
        out = next(iter(self.outputs.values()))
        return out.get_buffer()  # returns a NumPy array
