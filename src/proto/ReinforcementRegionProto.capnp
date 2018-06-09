@0xdb8c604b986c1aa5;

# Link to nupic proto schema
using import "/proto/ExtendedTemporalMemoryProto.capnp".ExtendedTemporalMemoryProto;

# Next ID: 4
struct ReinforcementRegionProto {
  # From ETM-Region
  implementation @0 :Text;
  learn @1 :Bool;

  # ETM Instance serialization
  etm @2 :ExtendedTemporalMemoryProto;

  # Reinforcement learning specific
  discount @3 :Float32;
  learningRate @4 :Float32;
  traceDecay @5 :Float32;
  stateValue @6 :Float32;

  # Lists of float values
  traces @7 :List(Float32);
  values @8 :List(Float32);

  # Lists of indices
  prevActiveCells @9 :List(UInt32);
  prevActiveCellsExternalBasal @10 :List(UInt32);
}
