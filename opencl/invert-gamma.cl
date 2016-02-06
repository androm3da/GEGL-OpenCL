__kernel void cl_invert_gamma(__global const float4 *in,
                              __global       float4 *out)
{
  ulong gid = get_global_id(0);
  const float4 in_v = in[gid];
  float4 out_v;
  out_v.xyz = 1.f - in_v.xyz;
  out_v.w = in_v.w;

  out[gid] = out_v;
}
