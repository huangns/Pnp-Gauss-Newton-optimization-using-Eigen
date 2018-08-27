//
// Created by h on 18-8-27.
//

#ifndef QQ_SE3_H
#define QQ_SE3_H

#include <iostream>
#include <cmath>
#include <limits>
#include <vector>

#include <Eigen/Dense>
#include <Eigen/Geometry>


using namespace Eigen;

struct TangentAndTheta {
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    Matrix<double,3,1> tangent;
    double theta;
};

class SE3 {
public:
    SE3(Quaterniond& q,Matrix<double ,3,1>& T,Matrix<double,6,1>& Lie);
    SE3(SE3& se3);
    SE3(Eigen::Matrix<double,6,1>& lie);
    SE3(Eigen::Matrix3d& R, Eigen::Matrix<double,3,1>& T);

    SE3& operator=(SE3&se3);


    Eigen::Quaternion<double > expAndTheta(Eigen::Vector3d const& omega, double* theta);
    Matrix3d hat(Eigen::Vector3d& omega);
    SE3 exp(Eigen::Matrix<double ,6,1> & a);
    void predictOutput(Eigen::VectorXd& input,Eigen::Matrix3d& K,Eigen::VectorXd& output);
    void computeJacobian(Eigen::VectorXd& input,Eigen::Matrix3d& K,Eigen::MatrixXd& jmat);
    void GN(Eigen::VectorXd& inputs,
            Eigen::VectorXd& measOutput,Eigen::Matrix3d& K);
    TangentAndTheta logAndTheta();
    void log();
    void update(Eigen::Matrix<double,6,1>& dL);


    Quaterniond q_;
    Vector3d T_;
    Eigen::Matrix<double,6,1> Lie_;
};


#endif //QQ_SE3_H
