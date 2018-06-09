@0xf8812f2ba77e13fa;

struct RandomProto {
  seed @0 :UInt64;
  impl @1 :RandomImplProto;
}

struct RandomImplProto {
  state @0 :List(Int64);
  rptr @1 :Int64;
  fptr @2 :Int64;
}
