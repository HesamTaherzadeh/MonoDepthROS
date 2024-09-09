import rclpy
from rclpy.node import Node
import rosbag2_py
from sensor_msgs.msg import PointCloud2
import pclpy
from pclpy import pcl
import sensor_msgs_py.point_cloud2 as pc2


class BagToPCDConverter(Node):
    def __init__(self):
        super().__init__('bag_to_pcd_converter')

    def read_bag_and_convert(self, bag_file, topic_name, output_pcd_file):
        # Initialize ROS 2 bag reader
        bag_reader = rosbag2_py.SequentialReader()

        # Specify the storage options and open the bag file
        storage_options = rosbag2_py.StorageOptions(uri=bag_file, storage_id='sqlite3')
        converter_options = rosbag2_py.ConverterOptions(input_serialization_format='cdr', output_serialization_format='cdr')
        bag_reader.open(storage_options, converter_options)

        # Get metadata from the bag
        topic_types = bag_reader.get_all_topics_and_types()
        type_map = {topic.name: topic.type for topic in topic_types}

        # Prepare for conversion
        cloud_points = []

        # Iterate through messages in the bag
        while bag_reader.has_next():
            (topic, data, timestamp) = bag_reader.read_next()
            if topic == topic_name:
                # Deserialize the PointCloud2 message
                msg = PointCloud2()
                msg.deserialize(data)

                # Convert PointCloud2 to list of points
                for point in pc2.read_points(msg, field_names=("x", "y", "z"), skip_nans=True):
                    cloud_points.append([point[0], point[1], point[2]])

        if not cloud_points:
            self.get_logger().error(f"No point cloud data found in {bag_file} on topic {topic_name}")
            return

        # Convert the list of points to a PCL point cloud object
        point_cloud = pcl.PointCloud.PointXYZ()
        point_cloud.from_array(cloud_points)

        # Save the point cloud to a PCD file
        pcl.io.savePCDFile(output_pcd_file, point_cloud)
        self.get_logger().info(f"PCD file saved to {output_pcd_file}")


def main(args=None):
    rclpy.init(args=args)

    # Specify your bag file, topic name, and output PCD file path here
    bag_file = '/home/hesam/Desktop/rosbag2_2024_09_09-12_14_40/rosbag2_2024_09_09-12_14_40_0.db3'
    topic_name = '/cloud_map'
    output_pcd_file = '/home/hesam/Desktop/output_point_cloud.pcd'

    converter_node = BagToPCDConverter()
    converter_node.read_bag_and_convert(bag_file, topic_name, output_pcd_file)

    rclpy.shutdown()


if __name__ == '__main__':
    main()
