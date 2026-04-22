import asyncio
import os
import time
import pickle
from typing import Any

import websockets

import rclpy
from rclpy.node import Node
from rclpy.subscription import Subscription
from rclpy.publisher import Publisher

from sensor_msgs.msg import JointState, CompressedImage
from rosidl_runtime_py.utilities import get_message

import sys
sys.path.append(os.path.normpath(f'{__file__}/../../'))
from python.utils.session import Session

subs: dict[str, Subscription] = { }
pubs: dict[str, Publisher] = { }

class RosSession(Session):
    def __init__(self, timeout=1200) -> None:
        self.node = Node('ros_bridge')
        self.sock: None | websockets.ClientConnection = None
        super().__init__(timeout)
    
    def on_call(self, method: str, args: tuple) -> Any:
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
        return super().on_call(method, args)
    
    def on_message(self, topic, msg: JointState | CompressedImage):
        self.msgs.put_nowait({ 'ret': { 'topic': topic, 'msg': msg } })
    
    def step_once(self):
        rclpy.spin_once(self.node)
        return super().step_once()

    async def connect(self):
        server_url = os.environ.get('WS_ADDR', 'ws://localhost:13001/api/ros/ws')
        async for sock in websockets.connect(server_url, max_size=None, ping_interval=20, ping_timeout=20):
            self.sock = sock
            async for data in sock:
                method, args, call = pickle.loads(data) # type: ignore
                res: dict = { 'call': call }
                try:
                    # TODO: 定位为什么走到这里就挂了
                    res['ret'] = await self.call(method, *args)
                except Exception as err:
                    res['err'] = err
                self.msgs.put_nowait(res)
    
    async def emit(self):
        while True:
            msg = await self.msgs.get()
            if self.sock:
                res = [msg.get('call'), msg.get('err'), msg.get('ret')]
                await self.sock.send(pickle.dumps(res), False)
    
    async def start(self):
        await asyncio.gather(self.connect(), self.emit())
    
if __name__ == '__main__':
    rclpy.init()
    sess = RosSession()
    asyncio.run(sess.start())
