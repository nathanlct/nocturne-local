"""Construct a scenarios.json file from a waymos protobuf
"""

import math
import json

from typing import Any, Dict, Iterator, Optional

import numpy as np
import tensorflow as tf

from waymo_open_dataset.protos import map_pb2, scenario_pb2

ERR_VAL = -1e4

_WAYMO_OBJECT_STR = {
    scenario_pb2.Track.TYPE_UNSET: "unset",
    scenario_pb2.Track.TYPE_VEHICLE: "vehicle",
    scenario_pb2.Track.TYPE_PEDESTRIAN: "pedestrian",
    scenario_pb2.Track.TYPE_CYCLIST: "cyclist",
    scenario_pb2.Track.TYPE_OTHER: "other",
}

_WAYMO_ROAD_STR = {
    scenario_pb2.DynamicMapState
}


def _parse_object_state(states: scenario_pb2.ObjectState, 
                        final_state: scenario_pb2.ObjectState) -> Dict[str, Any]:
    return {
        "position": {
            "x": [state.center_x if state.valid else ERR_VAL for state in states ],
            "y": [state.center_y if state.valid else ERR_VAL for state in states ],
        },
        "width": states[0].width,
        "length": states[0].length,
        "heading": [math.degrees(state.heading) if state.valid else ERR_VAL for state in states],  # Use rad here?
        "velocity": {
            "x": [state.velocity_x if state.valid else ERR_VAL for state in states],
            "y": [state.velocity_y if state.valid else ERR_VAL for state in states],
        },
        "valid": [state.valid for state in states],
        "goalPosition": {
            "x": final_state.center_x,
            "y": final_state.center_y
        }
    }


def _init_object(track: scenario_pb2.Track) -> Optional[Dict[str, Any]]:
    final_valid_index = 0
    # TODO(eugenevinitsky) valid may be false at the start
    for i, state in enumerate(track.states):
        if state.valid:
            final_valid_index = i

    obj = _parse_object_state(track.states, track.states[final_valid_index])
    obj["type"] = _WAYMO_OBJECT_STR[track.object_type]
    return obj


def _init_road(map_feature: map_pb2.MapFeature) -> Optional[Dict[str, Any]]:
    feature = map_feature.WhichOneof("feature_data")
    if feature == 'stop_sign':
        p = getattr(map_feature, map_feature.WhichOneof("feature_data")).position
        geometry = [{"x": p.x, "y": p.y}]
    elif feature != 'crosswalk' and feature != 'speed_bump':
        try:
            geometry = [{"x": p.x, "y": p.y} for p in getattr(map_feature, map_feature.WhichOneof("feature_data")).polyline]
        except:
            return None
    else:
        geometry = [{"x": p.x, "y": p.y} for p in getattr(map_feature, map_feature.WhichOneof("feature_data")).polygon]
    return {
        "geometry": geometry,
        "type": map_feature.WhichOneof("feature_data"),
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
        "name": scenario_path.split('/')[-1],
        "objects": objects,
        "roads": roads,
    }
    with open(scenario_path, "w") as f:
        json.dump(scenario, f)