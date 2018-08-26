#ifndef SE_H
#define SE_H

#include <iostream>
#include <cmath>
#include <limits>
#include <vector>
#include "./Eigen/Dense"
#include "./Eigen/Geometry"
using namespace std;
using namespace Eigen;

namespace SE3_Optimization
{
	using namespace Eigen;

	struct TangentAndTheta {
		EIGEN_MAKE_ALIGNED_OPERATOR_NEW
		Matrix<double,3,1> tangent;
		double theta;
	};


	class SE3
	{
	public:

		SE3(Quaterniond& q,Matrix<double ,3,1>& T,Matrix<double,6,1>& Lie):q_(q),T_(T),Lie_(Lie)
		{

		}
		SE3(SE3& se3):q_(se3.q_),T_(se3.T_),Lie_(se3.Lie_)
		{

		}
		SE3(Eigen::Matrix<double,6,1>& lie):Lie_(lie)
		{
			
			SE3 se3=exp(Lie_);
			q_=se3.q_;
			T_=se3.T_;
		}

		Quaterniond q_;
		Vector3d T_;
		Eigen::Matrix<double,6,1> Lie_;

		Eigen::Quaternion<double > expAndTheta(Eigen::Vector3d const& omega, double* theta)//李代数旋转部分向四元数的转换
		{
			using std::abs;
			using std::cos;
			using std::sin;
			using std::sqrt;

			double theta_sq = omega.squaredNorm();
			*theta = sqrt(theta_sq);
			double half_theta = (0.5) * (*theta);

			double imag_factor;
			double real_factor;
			if( (*theta) < std::numeric_limits<double >::epsilon() )
			{
				double theta_po4 = theta_sq * theta_sq;
				imag_factor = (0.5) - (1.0 / 48.0) * theta_sq +
					(1.0 / 3840.0) * theta_po4;
				real_factor = (1) - (1.0 / 8.0) * theta_sq +
					(1.0 / 384.0) * theta_po4;
			} else {
				double sin_half_theta = sin(half_theta);
				imag_factor = sin_half_theta / (*theta);
				real_factor = cos(half_theta);
			}

			Eigen::Quaterniond q(real_factor, imag_factor * omega.x(),
				imag_factor * omega.y(), imag_factor * omega.z());
			q.normalize();
			return q;
		}




		Matrix3d hat(Eigen::Vector3d& omega)
		{
			Matrix3d M;
			M<<0,-omega.z(),omega.y(),
				omega.z(),0,-omega.x(),
				-omega.y(),omega.x(),0;
			return  M;
		}

		SE3  exp(Matrix<double ,6,1> & a)//李代数向四元数和平移向量之间的转换
		{
			using std::cos;
			using std::sin;
			//Vector3<Scalar> const omega = a.template tail<3>();
			Vector3d omega=a.block<3,1>(3,0);
			double theta;
			Quaterniond q(expAndTheta(omega, &theta));
			Matrix3d const Omega = hat(omega);
			Matrix3d const Omega_sq = Omega * Omega;
			Matrix3d V;

			if (theta < std::numeric_limits<double >::epsilon())
			{
				V.setIdentity();
				// Note: That is an accurate expansion!
			}
			else
			{
				double theta_sq = theta * theta;
				V = (Matrix3d::Identity() +
					(1 - cos(theta)) / (theta_sq)*Omega +
					(theta - sin(theta)) / (theta_sq * theta) * Omega_sq);
			}
			Matrix<double ,3,1> tmpm=V*a.block<3,1>(0,0);

			SE3 tmp(q, tmpm,a);
			return tmp;
		}

		//void predictOutput(std::vector<Eigen::Matrix<double ,3,1> >& input, SE3& se3,Eigen::Matrix3d& K,std::vector<Eigen::Vector2d > & output)
		//void predictOutput(std::vector<Eigen::Matrix<double ,3,1> >& input, SE3& se3,Eigen::Matrix3d& K,Eigen::VectorXd& output)
		void predictOutput(Eigen::VectorXd& input,Eigen::Matrix3d& K,Eigen::VectorXd& output)
		{
			//EIGEN_MAKE_ALIGNED_OPERATOR_NEW
			Eigen::Matrix3d R=q_.conjugate().toRotationMatrix();
			Eigen::Matrix<double,3,1> t=T_;
			int inputSize=input.size();
			double fx=K(0,0);
			double cx=K(0,2);
			double fy=K(1,1);
			double cy=K(1,2);
			output.resize(inputSize);
			for (int i = 0; i < inputSize; ++i) {
				Eigen::Matrix<double,3,1> Pc=R*input.block<3,1>(3*i,0) +t;
				double ux=fx*Pc(0,0)/Pc(2,0)+cx;
				double uy=fy*Pc(1,0)/Pc(2,0)+cy;
				Eigen::Matrix<double,2,1> tempM;
				tempM(0,0)=ux;
				tempM(1,0)=uy;

				//tempM<<ux,uy;
				//output[i]=(tempM);
				output(2*i,0)=ux;
				output(2*i+1,0)=uy;
			}
		}


		//void computeJacobian(std::vector<Eigen::Matrix<double ,3,1> >& input, SE3& se3,Eigen::Matrix3d& K,Eigen::MatrixXd& jmat)
		//void computeJacobian(Eigen::VectorXd& input, SE3& se3,Eigen::Matrix3d& K,Eigen::MatrixXd& jmat)
		void computeJacobian(Eigen::VectorXd& input,Eigen::Matrix3d& K,Eigen::MatrixXd& jmat)
		{
			//int inputSize=input.size();
			int inputSize=input.rows()/3;
			Eigen::Matrix3d R=q_.conjugate().toRotationMatrix();
			Eigen::Matrix<double,3,1> t=T_;
			double fx=K(0,0);
			double cx=K(0,2);
			double fy=K(1,1);
			double cy=K(1,2);
			for (int i = 0; i < inputSize; ++i) {
				Eigen::Matrix<double ,3,1> Pc=R*input.block<3,1>(3*i,0)+t;
				double X=Pc(0,0);
				double Y=Pc(1,0);
				double Z=Pc(2,0);
				jmat(2*i,0)=fx*X*Y/(Z*Z);
				jmat(2*i,1)=-(fx+fx*X*X/(Z*Z));
				jmat(2*i,2)=fx*Y/Z;
				jmat(2*i,3)=-fx/Z;
				jmat(2*i,4)=0;
				jmat(2*i,5)=fx*X/(Z*Z);

				jmat(2*i+1,0)=fy+fy*Y*Y/(Z*Z);
				jmat(2*i+1,1)=-fy*X*Y/(Z*Z);
				jmat(2*i+1,2)=-fy*X/Z;
				jmat(2*i+1,3)=0;
				jmat(2*i+1,4)=-fy/Z;
				jmat(2*i+1,5)=fy*Y/(Z*Z);
			}
		}

		void GN(Eigen::VectorXd& inputs,
			Eigen::VectorXd& measOutput,Eigen::Matrix3d& K)
		{
			int m = measOutput.rows();
			//int n = params.rows();
			// jacobian 
			//MatrixXd jmat(m, n);
			VectorXd r(m, 1);
			//VectorXd tmp(m, 1);
			//int inputsize=inputs.size();
			Eigen::MatrixXd JMat(m,6);
			Eigen::VectorXd computeOutput(m,1);
			//Eigen::VectorXd r(2*inputsize,1);
			for (int i=0;i<10;++i)
			{
				predictOutput(inputs, K,computeOutput);
				r=measOutput-computeOutput;
				computeJacobian(inputs,K,JMat);
				Matrix<double,6,1> deltaX=(JMat.transpose()*JMat).inverse()*(-JMat.transpose()*r);
				update(deltaX);
				//predictOutput(input, ,Eigen::Matrix3d& K,Eigen::VectorXd& output)
			}

		}

		//返回theta,和旋转向量theta*(nx,ny,nz)
		TangentAndTheta logAndTheta()  {
			TangentAndTheta J;
			using std::abs;
			using std::atan;
			using std::sqrt;
			double squared_n = q_.vec().squaredNorm();// .vec().squaredNorm();
			double n = sqrt(squared_n);
			double w = q_.w();

			double two_atan_nbyw_by_n;

			// Atan-based log thanks to
			//
			// C. Hertzberg et al.:
			// "Integrating Generic Sensor Fusion Algorithms with Sound State
			// Representation through Encapsulation of Manifolds"
			// Information Fusion, 2011

			if (n < std::numeric_limits<double >::epsilon()) {
				// If quaternion is normalized and n=0, then w should be 1;
				// w=0 should never happen here!
				//SOPHUS_ENSURE(abs(w) >= Constants<Scalar>::epsilon(),
				//	"Quaternion (%) should be normalized!",
				//	unit_quaternion().coeffs().transpose());
				double squared_w = w * w;
				two_atan_nbyw_by_n =
					2.0 / w - 2 * (squared_n) / (w * squared_w);
			} else {
				if (abs(w) <std::numeric_limits<double >::epsilon()) {
					if (w > 0) {
						two_atan_nbyw_by_n = 3.14159265358979323846 / n;
					} else {
						two_atan_nbyw_by_n = -3.14159265358979323846 / n;
					}
				} else {
					two_atan_nbyw_by_n = 2.0 * atan(n / w) / n;
				}
			}

			J.theta = two_atan_nbyw_by_n * n;

			J.tangent = two_atan_nbyw_by_n * q_.vec();
			return J;
		}

		//Matrix<double,6,1> log()
		void log()
		{
			typedef Matrix<double,6,1> Tangent;
			using std::abs;
			using std::cos;
			using std::sin;
			Tangent upsilon_omega;
			TangentAndTheta omega_and_theta = logAndTheta();
			double theta = omega_and_theta.theta;
			upsilon_omega.block<3,1>(3,0) = omega_and_theta.tangent;
			Matrix3d  Omega =hat(omega_and_theta.tangent);

			if (abs(theta) < std::numeric_limits<double>::epsilon()) {
				Matrix3d V_inv = Matrix3d::Identity() -
					(0.5) * Omega +
					(1. / 12.) * (Omega * Omega);

				upsilon_omega.block<3,1>(0,0) = V_inv *T_;
			} else {
				double half_theta = 0.5 * theta;

				Matrix3d V_inv =
					(Matrix3d::Identity() - 0.5 * Omega +
					(1 -
					theta * cos(half_theta) / (2 * sin(half_theta))) /
					(theta * theta) * (Omega * Omega));
				upsilon_omega.template head<3>() = V_inv * T_;
			}
			Lie_=upsilon_omega;
			//return upsilon_omega;
		}
		void update(Eigen::Matrix<double,6,1>& dL)
		{
			SE3 tempSE=exp(dL);
			Matrix4d Td;
			Td.setIdentity();
			Td.block<3,3>(0,0)=tempSE.q_.conjugate().toRotationMatrix();
			Td.block<3,1>(0,3)=tempSE.T_;

			Matrix4d Tnow;
			Tnow.setIdentity();
			Tnow.block<3,3>(0,0)=q_.conjugate().toRotationMatrix();
			Tnow.block<3,1>(0,3)=T_;

			Matrix4d Tupdate=Td*Tnow;
			q_=Eigen::Quaterniond((Tupdate.block<3,3>(0,0)).transpose());
			T_=Tupdate.block<3,1>(0,0);
			log();
		}
	};	
}
#endif
