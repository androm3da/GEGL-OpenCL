__kernel void cl_invert_gamma(__global const float4 *in,
                              __global       float4 *out)
{
  ulong gid = get_global_id(0);
  const float4 in_v = in[gid];
  const float4 out_v =
                {
                    1.f - in_v.x,
                    1.f - in_v.y,
                    1.f - in_v.z,
                    in_v.w};
  out[gid] = out_v;
}
