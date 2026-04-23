import asyncio
import os
import queue
import time
import pickle
import traceback
from typing import Any

import websockets

import rclpy                                             # type: ignore
from rclpy.node import Node                              # type: ignore
from rclpy.subscription import Subscription              # type: ignore
from rclpy.publisher import Publisher                    # type: ignore

from sensor_msgs.msg import JointState, CompressedImage  # type: ignore

from rosidl_runtime_py.utilities import get_message      # type: ignore
from rosidl_runtime_py import message_to_ordereddict     # type: ignore

import sys
sys.path.append(os.path.normpath(f'{__file__}/../../'))
from python.utils.session import Session

subs: dict[str, Subscription] = { }
pubs: dict[str, Publisher] = { }

class RosSession(Session):
    def __init__(self) -> None:
        self.node = Node('ros_bridge')
        self.sock: None | websockets.ClientConnection = None
        super().__init__(0)
    
    def call_once(self, method: str, args: tuple) -> Any:
        if method == 'list_topics':
            return self.node.get_topic_names_and_types()
        elif method == 'subscribe' or method == 'publish':
            topic, type = args
            msg_type = get_message(type)
            if method == 'subscribe':
                if not topic in subs:
                    callback = lambda msg, topic=topic: self.on_message(topic, msg)
                    subs[topic] = self.node.create_subscription(msg_type, topic, callback, 10)
            else:
                if not topic in pubs:
                    pubs[topic] = self.node.create_publisher(msg_type, topic, 10)
                if type == 'JointState':
                    msg = JointState()
                elif type == 'CompressedImage':
                    msg = CompressedImage()
                else:
                    raise Exception(f'unknown message type {type}')
                pubs[topic].publish(msg)
        return super().call_once(method, args)
    
    def on_message(self, topic, msg):
        msg = message_to_ordereddict(msg)
        self.msgs.put_nowait({ 'topic': topic, 'msg': msg })
    
    def step_once(self):
        rclpy.spin_once(self.node, timeout_sec=0.01)
        return super().step_once()

    async def connect_once(self):
        server_url = os.environ.get('WS_ADDR', 'ws://localhost:13001/api/ros/ws')
        async for sock in websockets.connect(server_url, max_size=None, ping_interval=20, ping_timeout=20):
            self.sock = sock
            async for data in sock:
                method, args, call = pickle.loads(data) # type: ignore
                err, ret = None, None
                try:
                    ret = self.call_once(method, args)
                except Exception as exception:
                    err = exception
                if self.sock:
                    await self.sock.send(pickle.dumps([call, err, ret]), False)

    async def connect(self):
        while True:
            try:
                await self.connect_once()
            except:
                traceback.print_exc()
                await asyncio.sleep(0.1)
    
    async def emit(self):
        while True:
            try:
                msg = self.msgs.get(False)
                if self.sock:
                    await self.sock.send(pickle.dumps(['', None, msg]), False)
            except queue.Empty:
                await asyncio.sleep(0.1)
    
    async def start(self):
        await asyncio.gather(self.connect(), self.emit())
    
if __name__ == '__main__':
    rclpy.init()
    sess = RosSession()
    asyncio.run(sess.start(), debug=True)
