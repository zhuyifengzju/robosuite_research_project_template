from easydict import EasyDict
import numpy as np
import random
from copy import deepcopy
from envs.base_domain import BaseDomain
from robosuite.models.objects import CylinderObject, BoxObject
from robosuite.utils.placement_samplers import UniformRandomSampler, SequentialCompositeSampler
from robosuite.utils.mjcf_utils import array_to_string

from envs.objects import PotObject, LShapeTool
from envs.utils import MultiRegionRandomSampler

from robosuite.models.arenas import TableArena
from robosuite.wrappers import DomainRandomizationWrapper


TOOL_DICT = {"material": "redwood",
             "xrange": [[0.06, 0.08]],
             "yrange": [[-0.23, -0.03]]}
POT_DICT = {"material": "steel",
            "loc": [0.0, 0.18]}
CUBE_DICT = {"material": "bluewood",
             "xrange": [[0.29, 0.32]],
             "yrange": [[-0.25, -0.10]],
             "size": [0.02, 0.025, 0.02]}
DISTRACT_DICT = {"num": 0,
                 "material_list": ['lightwood', 'darkwood', 'redwood'],
                 "xrange": [[0.30, 0.34]],
                 "yrange": [[0.18, 0.24]]}

TABLE_DICT = {"xml_file": "../assets/normal_table.xml"}

TOOL_USE_SPECS = EasyDict({"pot": POT_DICT,
                           "tool": TOOL_DICT,
                           "cube": CUBE_DICT,
                           "distracting_objects": DISTRACT_DICT,
                           "table": TABLE_DICT})

def get_tool_use_exp_tasks(exp_name="normal", *args, **kwargs):
    task_specs = TOOL_USE_SPECS
    if exp_name == "normal":
        return ToolUseDomain(task_specs=task_specs, *args, **kwargs)

class ToolUseDomain(BaseDomain):
    def __init__(
            self,
            task_specs=TOOL_USE_SPECS,
            *args,
            **kwargs):
        self.task_specs = task_specs
        # if "table_xml_file" in kwargs:
        kwargs["table_xml_file"] = task_specs["table"]["xml_file"]
        super().__init__(*args, **kwargs)

    def _load_fixtures_in_arena(self, mujoco_arena):
        """In Pick Place domain, we do not have environment fixtures"""
        pass
        
    def _load_objects_in_arena(self, mujoco_arena):
        # Load environment fixtures

        self.objects_dict["pot"] = PotObject(
            name="pot",
            material=deepcopy(self.custom_material_dict[self.task_specs["pot"]["material"]])
        )

        self.objects_dict["tool"] = LShapeTool(
            name="tool",
            material=deepcopy(self.custom_material_dict[self.task_specs["tool"]["material"]])
        )

        cube_size = self.task_specs["cube"]["size"]
        self.objects_dict["cube"] = BoxObject(
            name="cube",
            size_min=cube_size,
            size_max=cube_size,
            rgba=[1, 0, 0, 1],
            material=self.custom_material_dict[self.task_specs["cube"]["material"]],
            density=500.,
        )
        

        if self.task_specs["distracting_objects"]["num"] > 0:
            for i in range(self.task_specs["distracting_objects"]["num"]):
                random_material_name = random.choice(self.task_specs["distracting_objects"]["material_list"])
                self.objects_dict[f"distracting_object_{i}"] = BoxObject(
                    name=f"distracting_object_{i}",
                    size=[0.02, 0.02, 0.02],
                    rgba=[1, 0, 0, 1],
                    material=self.custom_material_dict[random_material_name],
                    density=500.,
                )

    def _setup_placement_initializer(self, mujoco_arena):
        """Function to define the placement"""
        self.placement_initializer = SequentialCompositeSampler(name="ObjectSampler")

        self.placement_initializer.append_sampler(
        sampler = MultiRegionRandomSampler(
            name="ObjectSampler-cube",
            mujoco_objects=self.objects_dict["cube"],
            x_ranges=self.task_specs["cube"]["xrange"],
            y_ranges=self.task_specs["cube"]["yrange"],
            rotation=(-np.pi / 2., -np.pi / 2.),
            rotation_axis='z',
            ensure_object_boundary_in_range=False,
            ensure_valid_placement=True,
            reference_pos=self.table_offset,
            z_offset=0.01,
        ))

        self.placement_initializer.append_sampler(
        sampler = MultiRegionRandomSampler(
            name="ObjectSampler-lshape",
            mujoco_objects=self.objects_dict["tool"],
            x_ranges=self.task_specs["tool"]["xrange"],
            y_ranges=self.task_specs["tool"]["yrange"],
            rotation=(0., 0.),
            rotation_axis='z',
            ensure_object_boundary_in_range=False,
            ensure_valid_placement=True,
            reference_pos=self.table_offset,
            z_offset=0.02,
        ))
        
        pot_pos_x, pot_pos_y = self.task_specs["pot"]["loc"]
        pot_object = self.objects_dict["pot"].get_obj(); pot_object.set("pos", array_to_string((pot_pos_x, pot_pos_y, self.table_offset[2] + 0.05)))

        if self.task_specs["distracting_objects"]["num"] > 0:
            for i in range(self.task_specs["distracting_objects"]["num"]):
                distracting_object_name = f"distracting_object_{i}"
                self.placement_initializer.append_sampler(
                    sampler=MultiRegionRandomSampler(
                    name=distracting_object_name,
                    mujoco_objects=self.objects_dict[distracting_object_name],
                    x_ranges=self.task_specs["distracting_objects"]["xrange"],
                    y_ranges=self.task_specs["distracting_objects"]["yrange"],
                    rotation=(-np.pi / 2., np.pi / 2.),
                    rotation_axis='z',
                    ensure_object_boundary_in_range=False,
                    ensure_valid_placement=False,
                    reference_pos=self.table_offset,
                    z_offset=0.1,
                    )
                )

    def _check_success(self):
        """
        Check if the food item is in the pot
        """
        pot_pos = self.sim.data.body_xpos[self.obj_body_id["pot"]]
        food_item_pos = self.sim.data.body_xpos[self.obj_body_id["cube"]]

        object_in_pot = self.check_contact(self.objects_dict["cube"], self.objects_dict["pot"]) and np.linalg.norm(pot_pos[:2] - food_item_pos[:2]) < self.objects_dict["pot"].pot_length * 3 / 4
        return object_in_pot


    def get_object_names(self):
        
        obs = self._get_observations()

        object_names = list(self.obj_body_id.keys())
        object_names += ["robot0_eef"]

        return object_names
