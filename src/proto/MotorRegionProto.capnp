@0x80e53ea3b26ba2ed;

# Link to nupic proto schema
using import "/proto/MotorTMProto.capnp".MotorTMProto;

# Next ID: 4
struct MotorRegionProto {
  learn @0 :Bool;

  # ETM Instance serialization
  motorTM @1 :MotorTMProto;
}
