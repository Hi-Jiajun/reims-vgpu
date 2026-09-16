#!/usr/bin/env ruby
# Guest rail driver: load a reviewed `.mtlb` fixture with `newLibraryWithData:`
# and drive it through the guest Metal stack.
#
# Usage, inside the guest (the booted macOS guest has no compiler and no working
# python3; Ruby 2.6 + Fiddle is what it does have):
#
#   ruby guest_mtlb_dispatch.rb [path/to.mtlb]      # load + dispatch, prints GATE3 lines
#   ruby guest_mtlb_dispatch.rb --payload-check [path/to.mtlb]
#
# `--payload-check` builds the payload and prints its class and size without
# touching Metal; it fails when the bytes did not survive the round trip. That
# check is not decoration: this driver's predecessor handed `newLibraryWithData:`
# an NSData from `[NSData dataWithContentsOfFile:]` (class
# `__NSCFData`/`_NSInlineData`) instead of a `dispatch_data_t`, and Metal
# segfaulted the guest while digesting it — see
# `kb/mtlb-from-data-is-dispatch-data.md` for the measurements and the A/B.
#
# Fiddle cannot build the MTLSize-by-value dispatch calls, so those two go
# through libffi (`ffi_prep_cif_var` + `ffi_call`) with a hand-built `ffi_type`,
# because both `dispatchThreadgroups:threadsPerThreadgroup:` and
# `dispatchThreads:threadsPerThreadgroup:` take two 24-byte `MTLSize` structs by
# value.

$stdout.sync = true
require 'fiddle'

VOIDP = Fiddle::TYPE_VOIDP
VOID = Fiddle::TYPE_VOID
INT = Fiddle::TYPE_INT
SIZE = Fiddle::TYPE_SIZE_T

OBJC = Fiddle.dlopen('/usr/lib/libobjc.A.dylib')
COREGRAPHICS = Fiddle.dlopen('/System/Library/Frameworks/CoreGraphics.framework/CoreGraphics')
METAL = Fiddle.dlopen('/System/Library/Frameworks/Metal.framework/Metal')
LIBFFI = Fiddle.dlopen('/usr/lib/libffi.dylib')
LIBDISPATCH = Fiddle.dlopen('/usr/lib/libSystem.B.dylib')

MSGSEND = OBJC['objc_msgSend']
SELREG = OBJC['sel_registerName']
GETCLASS = OBJC['objc_getClass']

SELS = {}

def selptr(name)
  SELS[name] ||= Fiddle::Function.new(SELREG, [VOIDP], VOIDP)
                             .call(Fiddle::Pointer["#{name}\0"])
end

def clsptr(name)
  Fiddle::Function.new(GETCLASS, [VOIDP], VOIDP)
                  .call(Fiddle::Pointer["#{name}\0"])
end

def send_msg(receiver, selector, ret_type, arg_types = [], *args)
  fn = Fiddle::Function.new(MSGSEND, [VOIDP, VOIDP] + arg_types, ret_type)
  fn.call(receiver, selptr(selector), *args)
end

def nsstr(str)
  send_msg(clsptr('NSString'), 'stringWithUTF8String:', VOIDP, [VOIDP],
           Fiddle::Pointer["#{str}\0"])
end

def nsstr_value(ptr)
  return nil if ptr.nil? || ptr.to_i.zero?
  raw = send_msg(ptr, 'UTF8String', VOIDP)
  return nil if raw.nil? || raw.to_i.zero?
  Fiddle::Pointer.new(raw.to_i).to_s
end

def nserror_value(err_slot)
  ptr = err_slot[0, 8].unpack1('Q')
  return nil if ptr.zero?
  ptr = Fiddle::Pointer.new(ptr)
  "#{nsstr_value(send_msg(ptr, 'domain', VOIDP))}##{send_msg(ptr, 'code', Fiddle::TYPE_LONG)}: " \
    "#{nsstr_value(send_msg(ptr, 'localizedDescription', VOIDP))}"
end

def err_slot
  slot = Fiddle::Pointer.malloc(8)
  slot[0, 8] = [0].pack('Q')
  slot
end

# --- libffi plumbing for the two MTLSize-by-value dispatch calls -------------

FFI_TYPE_STRUCT = 13
FFI_UNIX64 = 2

def ffi_struct_of_three_uint64
  uint64 = LIBFFI['ffi_type_uint64']
  elements = Fiddle::Pointer.malloc(4 * 8)
  elements[0, 32] = [uint64, uint64, uint64, 0].pack('Q4')
  type = Fiddle::Pointer.malloc(24)
  type[0, 24] = [0].pack('Q') + [0].pack('S') + [FFI_TYPE_STRUCT].pack('S') +
                [0].pack('L') + [elements.to_i].pack('Q')
  [type, elements]
end

def build_cif
  struct_type, elements = ffi_struct_of_three_uint64
  pointer_type = LIBFFI['ffi_type_pointer']
  atypes = Fiddle::Pointer.malloc(5 * 8)
  atypes[0, 40] = [pointer_type, pointer_type, struct_type.to_i, struct_type.to_i, 0].pack('Q5')
  cif = Fiddle::Pointer.malloc(160)
  cif[0, 160] = "\0" * 160
  prep = Fiddle::Function.new(LIBFFI['ffi_prep_cif_var'],
                              [VOIDP, INT, INT, INT, VOIDP, VOIDP], INT)
  status = prep.call(cif, FFI_UNIX64, 2, 4, pointer_type, atypes)
  raise "ffi_prep_cif_var status=#{status}" unless status.zero?
  [cif, struct_type, elements, atypes]
end

def call_two_sizes(receiver, selector, size_a, size_b)
  cif, struct_type, elements, atypes = build_cif
  args = Fiddle::Pointer.malloc(8 + 8 + 24 + 24)
  args[0, 64] = [receiver.to_i, selptr(selector).to_i].pack('Q2') +
                size_a.pack('Q3') + size_b.pack('Q3')
  avalue = Fiddle::Pointer.malloc(4 * 8)
  base = args.to_i
  avalue[0, 32] = [base, base + 8, base + 16, base + 40].pack('Q4')
  rvalue = Fiddle::Pointer.malloc(8)
  Fiddle::Function.new(LIBFFI['ffi_call'], [VOIDP, VOIDP, VOIDP, VOIDP], VOID)
                 .call(cif, MSGSEND, rvalue, avalue)
  # Keep the type graph alive until after the call.
  [cif, struct_type, elements, atypes, args, avalue, rvalue]
end

# --- payload ------------------------------------------------------------------

DEFAULT_PATH = '/tmp/gate3_mul3add1.mtlb'
MODE = ARGV[0] == '--payload-check' ? :payload_check : :dispatch
PATH = (MODE == :payload_check ? ARGV[1] : ARGV[0]) || DEFAULT_PATH
RAW = File.binread(PATH)

def dispatch_data_from(bytes)
  # DISPATCH_DATA_DESTRUCTOR_DEFAULT (0) + NULL queue: libdispatch copies the
  # buffer, which is what the API wants (the library keeps the data alive).
  dd = Fiddle::Function.new(LIBDISPATCH['dispatch_data_create'],
                            [VOIDP, SIZE, VOIDP, VOIDP], VOIDP)
                       .call(Fiddle::Pointer[bytes], bytes.bytesize, 0, 0)
  size = Fiddle::Function.new(LIBDISPATCH['dispatch_data_get_size'], [VOIDP], SIZE).call(dd)
  raise "dispatch_data size #{size} != #{bytes.bytesize}" unless size == bytes.bytesize
  dd
end

if MODE == :payload_check
  dd = dispatch_data_from(RAW)
  klass = nsstr_value(send_msg(send_msg(dd, 'class', VOIDP), 'description', VOIDP))
  puts "PAYLOAD ok bytes=#{RAW.bytesize} class=#{klass.inspect} path=#{PATH}"
  exit 0
end

# --- driver -------------------------------------------------------------------

N_WORDS = 4
EXPECT = [4, 7, 10, 13].freeze

def readback(buffer)
  contents = send_msg(buffer, 'contents', VOIDP)
  Fiddle::Pointer.new(contents.to_i)[0, N_WORDS * 4].unpack('L<4')
end

def stage_input(buffer)
  contents = send_msg(buffer, 'contents', VOIDP)
  Fiddle::Pointer.new(contents.to_i)[0, N_WORDS * 4] = [1, 2, 3, 4].pack('L<4')
end

def run_arm(dev, pso, buffer, arm)
  cmd = send_msg(dev, 'newCommandQueue', VOIDP)
  cmd = send_msg(cmd, 'commandBuffer', VOIDP)
  enc = send_msg(cmd, 'computeCommandEncoder', VOIDP)
  send_msg(enc, 'setComputePipelineState:', VOID, [VOIDP], pso)
  send_msg(enc, 'setBuffer:offset:atIndex:', VOID, [VOIDP, SIZE, SIZE], buffer, 0, 0)
  if arm == :threadgroups
    call_two_sizes(enc, 'dispatchThreadgroups:threadsPerThreadgroup:', [1, 1, 1], [4, 1, 1])
  else
    call_two_sizes(enc, 'dispatchThreads:threadsPerThreadgroup:', [4, 1, 1], [4, 1, 1])
  end
  send_msg(enc, 'endEncoding', VOID)
  send_msg(cmd, 'commit', VOID)
  send_msg(cmd, 'waitUntilCompleted', VOID)
  status = send_msg(cmd, 'status', SIZE)
  err = send_msg(cmd, 'error', VOIDP)
  detail = nsstr_value(send_msg(err, 'localizedDescription', VOIDP)) if err && !err.to_i.zero?
  words = readback(buffer)
  puts format('GATE3 arm=%s dispatch=%s status=%d error=%s readback=%s correct=%s',
              arm, arm == :threadgroups ? 'dispatchThreadgroups' : 'dispatchThreads',
              status, detail.inspect, words.inspect, (words == EXPECT).inspect)
  [status == 4, words == EXPECT]
end

def main
  dev = Fiddle::Function.new(METAL['MTLCreateSystemDefaultDevice'], [], VOIDP).call
  return 1 if dev.nil? || dev.to_i.zero?
  puts "GATE3 device=ok name=#{nsstr_value(send_msg(dev, 'name', VOIDP)).inspect}"

  data = dispatch_data_from(RAW)
  puts "GATE3 payload=dispatch_data bytes=#{RAW.bytesize} class=OS_dispatch_data " \
       "sha256=#{`shasum -a 256 #{PATH}`.split.first}"
  err = err_slot
  lib = send_msg(dev, 'newLibraryWithData:error:', VOIDP, [VOIDP, VOIDP], data, err)
  if lib.nil? || lib.to_i.zero?
    puts "GATE3 newLibraryWithData FAILED: #{nserror_value(err).inspect}"
    return 1
  end
  names = send_msg(lib, 'functionNames', VOIDP)
  count = send_msg(names, 'count', SIZE)
  listed = (0...count).map do |i|
    nsstr_value(send_msg(names, 'objectAtIndex:', VOIDP, [SIZE], i))
  end
  puts "GATE3 library ok functions=#{listed.inspect}"
  entry = listed.include?('apv_cs') ? 'apv_cs' : listed.first
  return 1 if entry.nil?
  fn = send_msg(lib, 'newFunctionWithName:', VOIDP, [VOIDP], nsstr(entry))
  err[0, 8] = [0].pack('Q')
  pso = send_msg(dev, 'newComputePipelineStateWithFunction:error:', VOIDP,
                 [VOIDP, VOIDP], fn, err)
  if pso.nil? || pso.to_i.zero?
    puts "GATE3 pipeline FAILED: #{nserror_value(err).inspect}"
    return 1
  end
  puts "GATE3 pipeline ok entry=#{entry}"
  buffer = send_msg(dev, 'newBufferWithLength:options:', VOIDP, [SIZE, SIZE],
                    N_WORDS * 4, 0)
  results = {}
  stage_input(buffer)
  results[:threadgroups] = run_arm(dev, pso, buffer, :threadgroups)
  stage_input(buffer)
  results[:threads] = run_arm(dev, pso, buffer, :threads)
  puts "GATE3 summary #{results.inspect}"
  (results.values.all? { |ok, exact| ok && exact }) ? 0 : 1
end

exit main
