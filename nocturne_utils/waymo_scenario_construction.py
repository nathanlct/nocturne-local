"""Construct a scenarios.json file from a waymos protobuf
"""

import math
import json

from typing import Any, Dict, Iterator, Optional

import numpy as np
import tensorflow as tf

from waymo_open_dataset.protos import map_pb2, scenario_pb2

_WAYMO_OBJECT_STR = {
    scenario_pb2.Track.TYPE_UNSET: "unset",
    scenario_pb2.Track.TYPE_VEHICLE: "vehicle",
    scenario_pb2.Track.TYPE_PEDESTRIAN: "pedestrian",
    scenario_pb2.Track.TYPE_CYCLIST: "cyclist",
    scenario_pb2.Track.TYPE_OTHER: "other",
}


def _parse_object_state(state: scenario_pb2.ObjectState) -> Dict[str, Any]:
    return {
        "position": {
            "x": state.center_x,
            "y": state.center_y,
        },
        "width": state.width,
        "length": state.length,
        "heading": math.degrees(state.heading),  # Use rad here?
        "velocity": {
            "x": state.velocity_x,
            "y": state.velocity_y,
        },
    }


def _init_object(track: scenario_pb2.Track) -> Optional[Dict[str, Any]]:
    obj = _parse_object_state(track.states[0])
    obj["type"] = _WAYMO_OBJECT_STR[track.object_type]
    return obj


def _init_road(map_feature: map_pb2.MapFeature) -> Optional[Dict[str, Any]]:
    if map_feature.WhichOneof("feature_data") != "lane":
        return None
    geometry = [{"x": p.x, "y": p.y} for p in map_feature.lane.polyline]
    return {
        "geometry": geometry,
        "lanes": 1,  # Put a default value here.
        "lanewidth": 40.0,  # Put a default value here.
    }


def load_protobuf(protobuf_path: str) -> Iterator[scenario_pb2.Scenario]:
    dataset = tf.data.TFRecordDataset(protobuf_path, compression_type="")
    for data in dataset:
        scenario = scenario_pb2.Scenario()
        scenario.ParseFromString(bytearray(data.numpy()))
        yield scenario


def get_actions_from_protobuf(protobuf: scenario_pb2.Scenario, veh_id: int,
                              time_step: int) -> Dict[str, Any]:
    obj = protobuf.tracks[veh_id]
    state = obj.states[time_step]
    assert state.valid
    ret = _parse_object_state(state)
    ret["type"] = _WAYMO_OBJECT_STR[obj.object_type]
    return ret


def waymo_to_scenario(scenario_path: str,
                      protobuf: scenario_pb2.Scenario) -> None:
    # read the protobuf file to get the right state

    # write the json file
    # construct the road geometries
    # place the initial position of the vehicles

    objects = []
    for track in protobuf.tracks:
        obj = _init_object(track)
        if obj is not None:
            objects.append(obj)
    roads = []
    for map_feature in protobuf.map_features:
        road = _init_road(map_feature)
        if road is not None:
            roads.append(road)

    scenario = {
        "objects": objects,
        "roads": roads,
    }
    with open(scenario_path, "w") as f:
        json.dump(scenario, f)
