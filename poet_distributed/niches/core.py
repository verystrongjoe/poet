# Copyright (c) 2020 Uber Technologies, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


class Niche:
    def rollout_batch(self, thetas, batch_size, random_state, eval=False):
        import numpy as np
        returns = np.zeros(batch_size)  # batch_size 개의 에피소드 리워드
        lengths = np.zeros(batch_size, dtype='int')  # batch_size 개의 에피소드 step

        # thetas 개수랑 batch size와 동일하게 맞춰야함
        for i, theta in enumerate(thetas):
            # rollout은 Niche를 상속받은 객체의 rollout 호출 (theta, random_state, eval)
            returns[i], lengths[i] = self.rollout(
                theta, random_state=random_state, eval=eval)

        return returns, lengths
