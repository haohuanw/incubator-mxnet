# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.

import mxnet as mx
from mxnet.test_utils import assert_almost_equal

def test_batch_norm_runs_correctly_with_fix_gamma():
    sym, arg_params, aux_params = mx.model.load_checkpoint(
        "/home/ANT.AMAZON.COM/haohuw/Documents/docker_workspace"
        "/fcn8s_mobilenet_v2_inference_trt_clipped", 10)

    sym_trt = sym.get_backend_symbol("TensorRT")
    sym_trt.save("deconv_optimized-symbol.json")

    mx.contrib.tensorrt.init_tensorrt_params(sym_trt, arg_params, aux_params)

    executor = sym_trt.simple_bind(ctx=mx.gpu(), data=(1, 3, 480, 640), grad_req='null',
                               force_rebind=True)
    executor.copy_params_from(arg_params, aux_params)

    #executor_trt = sym_trt.simple_bind(ctx=mx.gpu(), data=(1, 1, 3, 3), grad_req='null',
    #                              force_rebind=True)
    #executor_trt.copy_params_from(arg_params_trt, aux_params_trt)

    #input_data = mx.nd.random.uniform(low=0, high=1, shape=(1, 1, 3, 3))

    #y = executor.forward(is_train=False, data=input_data)
    #y_trt = executor_trt.forward(is_train=False, data=input_data)

    #print(y[0].asnumpy())
    #print(y_trt[0].asnumpy())
    #assert_almost_equal(y[0].asnumpy(), y_trt[0].asnumpy(), 1e-4, 1e-4)

if __name__ == '__main__':
    import nose
    nose.runmodule()
