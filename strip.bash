sed -i '/locations_array__/d' $1
sed -i '/static constexpr std::array<const char[^;]*;/d' $1

sed '/^static constexpr std::array<const char[/,/^\$/{/;/!{/^\$/!d}}' 




