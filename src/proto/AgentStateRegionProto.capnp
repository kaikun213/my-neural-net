@0xa9703bedc0fe10d0;

# Link to nupic proto schema
using import "/proto/ExtendedTemporalMemoryProto.capnp".ExtendedTemporalMemoryProto;

# Next ID: 4
struct AgentStateRegionProto {
  # From ETM-Region
  implementation @0 :Text;
  learn @1 :Bool;

  # ETM Instance serialization
  etm @2 :ExtendedTemporalMemoryProto;

  # Lists of indices
  prevBasalInput @3 :List(UInt32);
}
