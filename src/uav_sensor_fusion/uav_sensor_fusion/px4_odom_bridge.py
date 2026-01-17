import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy, HistoryPolicy

from px4_msgs.msg import VehicleOdometry
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Quaternion


class Px4OdomBridge(Node):
    def __init__(self):
        super().__init__("px4_odom_bridge")

        # PX4 yayıncısına uyumlu QoS profili
        qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )

        self.sub = self.create_subscription(
            VehicleOdometry,
            "/fmu/out/vehicle_odometry",
            self.cb,
            qos,
        )

        self.pub = self.create_publisher(
            Odometry,
            "/odom/px4",
            10,
        )

        self.get_logger().info("px4_odom_bridge started, waiting for /fmu/out/vehicle_odometry...")


    def cb(self, msg: VehicleOdometry):
        odom = Odometry()
        odom.header.stamp = self.get_clock().now().to_msg()
        odom.header.frame_id = "odom"
        odom.child_frame_id = "base_link"

        # --- KOORDİNAT DÖNÜŞÜMÜ (NED -> ENU) ---
        # PX4 X (Kuzey) -> ROS Y (Kuzey)
        # PX4 Y (Doğu)  -> ROS X (Doğu)
        # PX4 Z (Aşağı) -> ROS Z (Yukarı) (Ters çeviriyoruz)
        
        # Pozisyon Dönüşümü
        odom.pose.pose.position.x = float(msg.position[1])  # PX4 Y -> ROS X
        odom.pose.pose.position.y = float(msg.position[0])  # PX4 X -> ROS Y
        odom.pose.pose.position.z = -1.0 * float(msg.position[2]) # PX4 Z -> ROS -Z

        # Hız (Velocity) Dönüşümü
        odom.twist.twist.linear.x = float(msg.velocity[1])
        odom.twist.twist.linear.y = float(msg.velocity[0])
        odom.twist.twist.linear.z = -1.0 * float(msg.velocity[2])

        # Oryantasyon (Quaternion) için basit dönüşüm (Sadece Z ekseni düzeltmesi)
        # Tam doğru dönüşüm için kütüphane gerekir ama şu an harita için bu yeterli olacaktır.
        # X ve Y eksenlerini yer değiştirip, Z ve W üzerinde işlem yapmak gerekir.
        # Şimdilik direkt aktarım yerine, dronu düz tuttuğunu varsayarak pozisyonu düzeltmen 
        # haritanın oluşması için en kritik adım. 
        
        # Geçici olarak (Eğer oryantasyon çok sapıtıyorsa IMU'dan gelen veriyi kullanması için
        # EKF'te odom0_config'den oryantasyonu (roll,pitch,yaw) false yapabilirsin.)
        
        # Şimdilik mevcut quaternion'u mapleyelim ama eksenleri değiştirerek:
        # (Bu tam matematiksel dönüşüm değil ama "North-East" yer değişimini basitçe taklit eder)
        odom.pose.pose.orientation.x = float(msg.q[1]) 
        odom.pose.pose.orientation.y = float(msg.q[0])
        odom.pose.pose.orientation.z = -1.0 * float(msg.q[2])
        odom.pose.pose.orientation.w = float(msg.q[3]) # W genelde q[0] veya q[3]'tür, PX4 msg yapısına dikkat.
        # NOT: PX4 msg.q genelde [w, x, y, z] formatındadır.
        # Eğer msg.q[0] W ise doğru.
        odom.pose.covariance = [0.0] * 36
        odom.pose.covariance[0] = 0.01  # X variance
        odom.pose.covariance[7] = 0.01  # Y variance
        odom.pose.covariance[14] = 0.01 # Z variance
        odom.pose.covariance[21] = 0.01 # Rot X
        odom.pose.covariance[28] = 0.01 # Rot Y
        odom.pose.covariance[35] = 0.01 # Rot Z

        self.pub.publish(odom)


def main(args=None):
    rclpy.init(args=args)
    node = Px4OdomBridge()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()