"""Verify the exact ROS Image buffer and benchmark the generated setter fast path."""
import ast
import array
import statistics
import time
import unittest
from pathlib import Path
import numpy as np
from sensor_msgs.msg import Image
from builtin_interfaces.msg import Time

source=Path(__file__).resolve().parents[1]/'sim_scene/run_feedback_sim.py'
tree=ast.parse(source.read_text())
function=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='to_img_msg')


class ImageTransportTests(unittest.TestCase):
    def test_buffer_and_stride_identical_for_rgb_and_depth(self):
        for data,encoding in [(np.arange(640*360*3,dtype=np.uint8).reshape(360,640,3),'bgr8'),
                              (np.linspace(0.,2.,640*360,dtype=np.float32).reshape(360,640),'32FC1')]:
            messages=[];timings=[]
            for optimized in [False,True]:
                namespace=dict(Image=Image,array=array,OPTIMIZED=optimized)
                exec(compile(ast.Module(body=[function],type_ignores=[]),str(source),'exec'),namespace)
                samples=[]
                for _ in range(3):
                    started=time.perf_counter()
                    msg=namespace['to_img_msg'](data,encoding,Time(sec=123,nanosec=456))
                    samples.append(time.perf_counter()-started)
                timings.append(statistics.median(samples));messages.append(msg)
            self.assertEqual(messages[0],messages[1])
            self.assertEqual(messages[1].data.tobytes(),data.tobytes())
            self.assertEqual(messages[1].step,data.strides[0])
            print(f'{encoding}: legacy={timings[0]:.6f}s buffer={timings[1]:.6f}s ratio={timings[0]/timings[1]:.1f}')


if __name__=='__main__':
    unittest.main()
