{
  // Convert a comma-delimited string to an array.
  toArray(str)::
    if std.type(str) == "string" && str != "null" && std.length(str) > 0 then
      std.split(str, ",")
    else [],

  nameValuePair(arg, kfield='name',
           vfield='value'):: 
  [
    { [kfield]: std.split(k,"=")[0], [vfield]: std.split(k,"=")[1]}
    for k in arg
  ],
}
