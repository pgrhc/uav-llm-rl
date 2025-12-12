# generated from rosidl_generator_py/resource/_idl.py.em
# with input from fusion_msgs:msg/FusionStamped.idl
# generated code does not contain a copyright notice


# Import statements for member types

import builtins  # noqa: E402, I100

import rosidl_parser.definition  # noqa: E402, I100


class Metaclass_FusionStamped(type):
    """Metaclass of message 'FusionStamped'."""

    _CREATE_ROS_MESSAGE = None
    _CONVERT_FROM_PY = None
    _CONVERT_TO_PY = None
    _DESTROY_ROS_MESSAGE = None
    _TYPE_SUPPORT = None

    __constants = {
    }

    @classmethod
    def __import_type_support__(cls):
        try:
            from rosidl_generator_py import import_type_support
            module = import_type_support('fusion_msgs')
        except ImportError:
            import logging
            import traceback
            logger = logging.getLogger(
                'fusion_msgs.msg.FusionStamped')
            logger.debug(
                'Failed to import needed modules for type support:\n' +
                traceback.format_exc())
        else:
            cls._CREATE_ROS_MESSAGE = module.create_ros_message_msg__msg__fusion_stamped
            cls._CONVERT_FROM_PY = module.convert_from_py_msg__msg__fusion_stamped
            cls._CONVERT_TO_PY = module.convert_to_py_msg__msg__fusion_stamped
            cls._TYPE_SUPPORT = module.type_support_msg__msg__fusion_stamped
            cls._DESTROY_ROS_MESSAGE = module.destroy_ros_message_msg__msg__fusion_stamped

            from builtin_interfaces.msg import Time
            if Time.__class__._TYPE_SUPPORT is None:
                Time.__class__.__import_type_support__()

            from fusion_msgs.msg import RadarPoints
            if RadarPoints.__class__._TYPE_SUPPORT is None:
                RadarPoints.__class__.__import_type_support__()

            from nav_msgs.msg import Odometry
            if Odometry.__class__._TYPE_SUPPORT is None:
                Odometry.__class__.__import_type_support__()

            from sensor_msgs.msg import Image
            if Image.__class__._TYPE_SUPPORT is None:
                Image.__class__.__import_type_support__()

            from sensor_msgs.msg import PointCloud2
            if PointCloud2.__class__._TYPE_SUPPORT is None:
                PointCloud2.__class__.__import_type_support__()

    @classmethod
    def __prepare__(cls, name, bases, **kwargs):
        # list constant names here so that they appear in the help text of
        # the message class under "Data and other attributes defined here:"
        # as well as populate each message instance
        return {
        }


class FusionStamped(metaclass=Metaclass_FusionStamped):
    """Message class 'FusionStamped'."""

    __slots__ = [
        '_stamp',
        '_image',
        '_lidar',
        '_radar',
        '_odom',
    ]

    _fields_and_field_types = {
        'stamp': 'builtin_interfaces/Time',
        'image': 'sensor_msgs/Image',
        'lidar': 'sensor_msgs/PointCloud2',
        'radar': 'fusion_msgs/RadarPoints',
        'odom': 'nav_msgs/Odometry',
    }

    SLOT_TYPES = (
        rosidl_parser.definition.NamespacedType(['builtin_interfaces', 'msg'], 'Time'),  # noqa: E501
        rosidl_parser.definition.NamespacedType(['sensor_msgs', 'msg'], 'Image'),  # noqa: E501
        rosidl_parser.definition.NamespacedType(['sensor_msgs', 'msg'], 'PointCloud2'),  # noqa: E501
        rosidl_parser.definition.NamespacedType(['fusion_msgs', 'msg'], 'RadarPoints'),  # noqa: E501
        rosidl_parser.definition.NamespacedType(['nav_msgs', 'msg'], 'Odometry'),  # noqa: E501
    )

    def __init__(self, **kwargs):
        assert all('_' + key in self.__slots__ for key in kwargs.keys()), \
            'Invalid arguments passed to constructor: %s' % \
            ', '.join(sorted(k for k in kwargs.keys() if '_' + k not in self.__slots__))
        from builtin_interfaces.msg import Time
        self.stamp = kwargs.get('stamp', Time())
        from sensor_msgs.msg import Image
        self.image = kwargs.get('image', Image())
        from sensor_msgs.msg import PointCloud2
        self.lidar = kwargs.get('lidar', PointCloud2())
        from fusion_msgs.msg import RadarPoints
        self.radar = kwargs.get('radar', RadarPoints())
        from nav_msgs.msg import Odometry
        self.odom = kwargs.get('odom', Odometry())

    def __repr__(self):
        typename = self.__class__.__module__.split('.')
        typename.pop()
        typename.append(self.__class__.__name__)
        args = []
        for s, t in zip(self.__slots__, self.SLOT_TYPES):
            field = getattr(self, s)
            fieldstr = repr(field)
            # We use Python array type for fields that can be directly stored
            # in them, and "normal" sequences for everything else.  If it is
            # a type that we store in an array, strip off the 'array' portion.
            if (
                isinstance(t, rosidl_parser.definition.AbstractSequence) and
                isinstance(t.value_type, rosidl_parser.definition.BasicType) and
                t.value_type.typename in ['float', 'double', 'int8', 'uint8', 'int16', 'uint16', 'int32', 'uint32', 'int64', 'uint64']
            ):
                if len(field) == 0:
                    fieldstr = '[]'
                else:
                    assert fieldstr.startswith('array(')
                    prefix = "array('X', "
                    suffix = ')'
                    fieldstr = fieldstr[len(prefix):-len(suffix)]
            args.append(s[1:] + '=' + fieldstr)
        return '%s(%s)' % ('.'.join(typename), ', '.join(args))

    def __eq__(self, other):
        if not isinstance(other, self.__class__):
            return False
        if self.stamp != other.stamp:
            return False
        if self.image != other.image:
            return False
        if self.lidar != other.lidar:
            return False
        if self.radar != other.radar:
            return False
        if self.odom != other.odom:
            return False
        return True

    @classmethod
    def get_fields_and_field_types(cls):
        from copy import copy
        return copy(cls._fields_and_field_types)

    @builtins.property
    def stamp(self):
        """Message field 'stamp'."""
        return self._stamp

    @stamp.setter
    def stamp(self, value):
        if __debug__:
            from builtin_interfaces.msg import Time
            assert \
                isinstance(value, Time), \
                "The 'stamp' field must be a sub message of type 'Time'"
        self._stamp = value

    @builtins.property
    def image(self):
        """Message field 'image'."""
        return self._image

    @image.setter
    def image(self, value):
        if __debug__:
            from sensor_msgs.msg import Image
            assert \
                isinstance(value, Image), \
                "The 'image' field must be a sub message of type 'Image'"
        self._image = value

    @builtins.property
    def lidar(self):
        """Message field 'lidar'."""
        return self._lidar

    @lidar.setter
    def lidar(self, value):
        if __debug__:
            from sensor_msgs.msg import PointCloud2
            assert \
                isinstance(value, PointCloud2), \
                "The 'lidar' field must be a sub message of type 'PointCloud2'"
        self._lidar = value

    @builtins.property
    def radar(self):
        """Message field 'radar'."""
        return self._radar

    @radar.setter
    def radar(self, value):
        if __debug__:
            from fusion_msgs.msg import RadarPoints
            assert \
                isinstance(value, RadarPoints), \
                "The 'radar' field must be a sub message of type 'RadarPoints'"
        self._radar = value

    @builtins.property
    def odom(self):
        """Message field 'odom'."""
        return self._odom

    @odom.setter
    def odom(self, value):
        if __debug__:
            from nav_msgs.msg import Odometry
            assert \
                isinstance(value, Odometry), \
                "The 'odom' field must be a sub message of type 'Odometry'"
        self._odom = value
