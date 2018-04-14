// renderer/ffi/vulkan.rs -- Aldaron's Device Interface / GPU / Vulkan
// Copyright (c) 2017-2018  Jeron A. Lau <jeron.lau@plopgrizzly.com>
// Licensed under the MIT LICENSE

use asi_vulkan;

pub struct Vulkan(pub asi_vulkan::Connection);

impl Vulkan {
	pub fn new(app_name: &str) -> Result<Self, String> {
		let connection = unsafe { asi_vulkan::load(app_name) };

		if connection.lib.is_null() {
			return Err("Failed to link to Vulkan.".to_string());
		}

		Ok(Vulkan(connection))
	}
}
